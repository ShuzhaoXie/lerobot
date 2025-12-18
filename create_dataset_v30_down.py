#!/usr/bin/env python
"""
Direct conversion from ABB raw data to LeRobot Dataset v3.0 format.
Combines logic from abb->v2.0 and v2.1->v3.0 conversion scripts.

Fixed version:
1. Correct stats.json format
2. Continuous episode indices (re-indexed after dropping invalid episodes)
3. Both action and state are 6D joint values (NO gripper)
"""

import os
import json
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset, Features, Image
import tqdm

# Constants for v3.0 format
CODEBASE_VERSION = "v3.0"
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 500
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file_{file_index:03d}.mp4"


# ✅ 修改1: 改为6维，去掉gripper
# OBS_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
# ACTION_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'ffmpeg'
    except FileNotFoundError:
        pass
    
    # Try common paths
    common_paths = [
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/opt/conda/bin/ffmpeg',
        shutil.which('ffmpeg')
    ]
    
    for path in common_paths:
        if path and Path(path).exists():
            logging.info(f"Found ffmpeg at: {path}")
            return path
    
    logging.error("FFmpeg not found. Please install ffmpeg:")
    logging.error("  Ubuntu/Debian: sudo apt-get install ffmpeg")
    logging.error("  Conda: conda install -c conda-forge ffmpeg")
    return None


FFMPEG_PATH = None


def get_video_info(video_path: Path) -> Dict:
    """Get video information using ffprobe."""
    ffprobe = str(FFMPEG_PATH).replace('ffmpeg', 'ffprobe') if FFMPEG_PATH != 'ffmpeg' else 'ffprobe'
    cmd = [
        ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames,width,height:format=duration',
        '-of', 'json', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    probe_data = json.loads(result.stdout)
    stream_data = probe_data['streams'][0]
    format_data = probe_data['format']
    fps_num, fps_den = map(int, stream_data['r_frame_rate'].split('/'))
    fps = fps_num / fps_den
    return {
        'duration': float(format_data.get('duration', 0.0)),
        'nb_frames': int(stream_data.get('nb_frames', '0')),
        'fps': fps,
        'width': int(stream_data['width']),
        'height': int(stream_data['height'])
    }


def run_ffmpeg_reencode(input_path: Path, output_path: Path, fps_rational: str) -> bool:
    """Re-encode video to h264, yuv420p, target fps and 448x448 resolution."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(FFMPEG_PATH), '-i', str(input_path),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            '-r', fps_rational,
            '-s', '448x448',
            '-y',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"FFmpeg error for {input_path.name}: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error running FFmpeg on {input_path.name}: {e}")
        return False


def concatenate_videos(video_paths: List[Path], output_path: Path) -> bool:
    """Concatenate multiple video files using ffmpeg."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file list for ffmpeg concat
        concat_file = output_path.parent / f"concat_list_{output_path.stem}.txt"
        with open(concat_file, 'w') as f:
            for vp in video_paths:
                f.write(f"file '{vp.absolute()}'\n")
        
        cmd = [
            str(FFMPEG_PATH), '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',
            '-y',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        concat_file.unlink()  # Clean up
        
        if result.returncode != 0:
            logging.error(f"FFmpeg concat error: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error concatenating videos: {e}")
        return False


def get_file_size_in_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


# ✅ 修改2: parse_action_states 只返回joint，不需要gripper了
def parse_joints(action_json_path: Path) -> List[Dict]:
    """Parse states list from action.json."""
    with open(action_json_path, 'r') as f:
        data = json.load(f)
    states = data.get('states', [])
    tt = data.get('time', [])
    result: List[Dict] = []
    
    for i, st in enumerate(states):
        if not isinstance(st, dict):
            result.append({
                'joint': None,
                'time': tt[i]
            })
            continue
        joint = st.get('joint')
        
        if joint is None or (not isinstance(joint, list)) or len(joint) != 6:
            result.append({
                'joint': None,
                'time': tt[i]
            })
            continue
            
        result.append({
            'joint': np.array(list(map(float, joint)), dtype=np.float32),
            'time': tt[i]
        })
    return result


def parse_joints_gripper(action_json_path: Path) -> List[Dict]:
    """Parse states list from action.json."""
    with open(action_json_path, 'r') as f:
        data = json.load(f)
    states = data.get('states', [])
    tt = data.get('time', [])
    # logging.info(f'times {len(tt)} {len(states)}')
    result: List[Dict] = []
    
    for i, st in enumerate(states):
        if not isinstance(st, dict):
            result.append({
                'joint': None,
                'gripper': None,
                'time': tt[i]
            })
            continue
        joint = st.get('joint')
        gripper = st.get('gripper')
        
        if joint is None or (not isinstance(joint, list)) or len(joint) != 6:
            result.append({
                'joint': None,
                'gripper': None,
                'time': tt[i]
            })
            continue
        
        # Parse gripper: "on" -> 1.0, "off" -> 0.0
        # TODO, maybe the "None" dose not mean the ``off gripper``
        if gripper is None:
            g = 0.0
        elif isinstance(gripper, str):
            g = 1.0 if gripper.lower() == 'on' else 0.0
        else:
            try:
                g = float(gripper)
            except Exception:
                g = 0.0
            
        result.append({
            'joint': np.array(list(map(float, joint)), dtype=np.float32),
            'gripper': g,
            'time': tt[i]
        })
    return result

def count_frames(path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames',
        '-of', 'json', str(path)
    ]
    output = subprocess.check_output(cmd).decode('utf-8')
    data = json.loads(output)
    return int(data['streams'][0]['nb_read_frames'])


def filter_and_save_video(src_path, dst_path, delete_ids, fps, sample_rate):
    frame_duration = 1.0 / float(fps)
    with tempfile.TemporaryDirectory(prefix="frames_") as frame_dir:
        print(f"创建临时目录: {frame_dir}")

        subprocess.run([
            "ffmpeg", "-i", src_path, 
            "-vf", "scale=448:448",
            "-vsync", "cfr", 
            "-qscale:v", "1", 
            f"{frame_dir}/%06d.png", 
            "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
        print(f"导出的帧数量: {len(files)} {files[0]}")

        if len(files) == 0:
            print("❗严重错误：没有帧被导出。视频本身可能有问题。")
            return
        
        total_frames = len(files)

        filtered_indices = [
            i for i in range(total_frames)
            if i not in delete_ids
        ]

        if not filtered_indices:
            print("❗过滤后没有任何帧可用。")
            return
        
        # -------------------------------------------------
        # Step 2️⃣：在过滤后的序列上按 sample_rate 采样
        # -------------------------------------------------
        if sample_rate > 1:
            sampled_indices = [
                filtered_indices[j]
                for j in range(len(filtered_indices))
                if j % sample_rate == 0
            ]
        else:
            sampled_indices = filtered_indices
        
        sampled_indices = sampled_indices[:-1] # remove the last frame, as we do not need the last obs
        
        # list_file = os.path.join(frame_dir, "file_list.txt")
        # with open(list_file, "w") as f:
        #     for i in range(total_frames):
        #         if i not in delete_ids:
        #             f.write(f"file '{frame_dir}/{(i+1):06d}.png'\n")
        #             f.write(f"duration {frame_duration:.8f}\n")

        list_file = os.path.join(frame_dir, "file_list.txt")
        with open(list_file, "w") as f:
            for i in sampled_indices:
                frame_path = f"{frame_dir}/{(i+1):06d}.png"
                f.write(f"file '{frame_path}'\n")
                f.write(f"duration {frame_duration:.8f}\n")
                
        subprocess.run([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "fast",
            "-y",
            dst_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        print(f"✅ 视频处理完成：{dst_path}")


class EpisodeData:
    """Container for episode data."""
    def __init__(self, episode_index: int, episode_dir: Path):
        self.episode_index = episode_index
        self.episode_dir = episode_dir
        self.states_data = []
        self.cam0_temp_path = None
        self.cam1_temp_path = None
        self.num_frames = 0
        self.fps = 0.0
        self.observations = []  # ✅ 6D joint state (no gripper)
        self.actions = []       # ✅ 6D joint action (no gripper)
        self.timestamps = []


def process_raw_episode(episode_dir: Path, episode_index: int, temp_dir: Path, fps_rational: str, sample_rate: int = 1, action_type: int = 0) -> EpisodeData:
    """Process a single raw episode directory."""
    ep_data = EpisodeData(episode_index, episode_dir)
    
    action_json = episode_dir / 'action.json'
    
    if not action_json.exists():
        logging.warning(f"Skipped {episode_dir.name}: missing action.json")
        return None
    
    cam0_in = None
    cam1_in = None
    for name in ['cam_0.mp4', 'gopro.mp4']:
        p = episode_dir / name
        if p.exists():
            cam0_in = p
            break
    for name in ['cam_1.mp4', 'kinect.mp4']:
        p = episode_dir / name
        if p.exists():
            cam1_in = p
            break
    
    if not cam0_in:
        logging.warning(f"Skipped {episode_dir.name}: missing cam0 video (gopro.mp4 or cam_0.mp4)")
        return None
    if not cam1_in:
        logging.warning(f"Skipped {episode_dir.name}: missing cam1 video (kinect.mp4 or cam_1.mp4)")
        return None
    
    try:
        if action_type == 0:
            raw_states_data = parse_joints_gripper(action_json)
        elif action_type == 1:
            raw_states_data = parse_joints(action_json)
        else:
            raise NotImplementedError
    except Exception as e:
        logging.warning(f"Skipped {episode_dir.name}: failed to parse action.json - {e}")
        return None
        
    if len(raw_states_data) == 0:
        logging.warning(f"Skipped {episode_dir.name}: no valid states in action.json")
        return None
    
    c0_count = count_frames(cam0_in)
    c1_count = count_frames(cam1_in)
    logging.info(f"c0count {c0_count}, c1count {c1_count}, lenraw {len(raw_states_data)}")
    try:
        assert c0_count == len(raw_states_data) and c1_count == c0_count
    except:
        logging.debug(f"name: {episode_dir.name}, count: {c0_count}, lenraw: {len(raw_states_data)}")
    
    states_data = []
    delete_ids = []
    
    if c0_count != len(raw_states_data):
        for i in range(len(raw_states_data), c0_count):
            delete_ids.append(i)
    
    # last_i = -1
    for i, st in enumerate(raw_states_data):
        if st["joint"] is None:
            delete_ids.append(i)
            continue
        else:
            states_data.append(st)
            # last_i = i
    # assert last_i != -1
    # delete_ids.append(last_i)
    delete_ids = list(set(sorted(delete_ids)))
    
    logging.info(f"delete_ids, {delete_ids}")
    logging.info(f'len(states_data) {len(states_data)}')
    
    # ✅ 修改3: 只提取joints，不需要grippers
    if action_type == 0:
        joints = [np.concatenate([state['joint'], [state['gripper']]]) for state in states_data]
    elif action_type == 1:
        joints = [state['joint'] for state in states_data]
    # downsample
    joints = joints[::sample_rate]
    # logging.info(f'len(joints) {len(joints)}')
    
    ep_data.cam0_temp_path = temp_dir / f'cam0_ep{episode_index:06d}.mp4'
    ep_data.cam1_temp_path = temp_dir / f'cam1_ep{episode_index:06d}.mp4'
    
    filter_and_save_video(cam0_in, ep_data.cam0_temp_path, delete_ids, fps_rational, sample_rate)
    filter_and_save_video(cam1_in, ep_data.cam1_temp_path, delete_ids, fps_rational, sample_rate)
    
    try:
        info0 = get_video_info(ep_data.cam0_temp_path)
        info1 = get_video_info(ep_data.cam1_temp_path)
    except Exception as e:
        logging.warning(f"Skipped {episode_dir.name}: failed to get video info - {e}")
        return None
    
    logging.info(f'joints: {len(joints)}')
    nf1 = info0.get('nb_frames', 0)
    nf2 = info1.get('nb_frames', 0)
    logging.info(f"nf1: {nf1}, nf2: {nf2}")
    
    ep_data.fps = info0.get('fps', fps_rational)
    ep_data.num_frames = len(joints) - 1
    logging.info(f"nf1: {nf1}, nf2: {nf2}, ep_data.num_frames {ep_data.num_frames}")
    if ep_data.num_frames != nf1:
        logging.info("----- no equal")
    
    timestamps = (np.arange(ep_data.num_frames, dtype=np.float32) / np.float32(ep_data.fps))
    
    # ✅ 修改4: 构建6D observation和action（不含gripper）
    for frame_idx in range(ep_data.num_frames):
        cur_state = joints[frame_idx].astype(np.float32) 
        cur_action = joints[frame_idx + 1].astype(np.float32)
        
        ep_data.observations.append(cur_state)
        ep_data.actions.append(cur_action)
        ep_data.timestamps.append(float(timestamps[frame_idx]))
    
    logging.info(f"✓ Processed {episode_dir.name} -> episode_{episode_index}: {ep_data.num_frames} frames @ {ep_data.fps:.2f} fps")
    return ep_data


def write_combined_data_file(episodes: List[EpisodeData], output_dir: Path, chunk_idx: int, file_idx: int, global_frame_offset: int):
    """Write combined data file for multiple episodes."""
    all_records = []
    
    for idx, ep_data in enumerate(episodes):
        for i in range(ep_data.num_frames):
            record = {
                'observation.state': ep_data.observations[i].tolist(),
                'action': ep_data.actions[i].tolist(),
                'episode_index': ep_data.episode_index,
                'frame_index': i,
                'timestamp': ep_data.timestamps[i],
                'index': global_frame_offset + i,
                'task_index': 0
            }
            all_records.append(record)
        global_frame_offset += ep_data.num_frames
    
    df = pd.DataFrame(all_records)
    path = output_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    
    return global_frame_offset


def write_combined_video_file(episodes: List[EpisodeData], output_dir: Path, camera_key: str, chunk_idx: int, file_idx: int):
    """Write combined video file for multiple episodes."""
    video_paths = []
    for ep_data in episodes:
        if 'cam_0' in camera_key:
            video_paths.append(ep_data.cam0_temp_path)
        else:
            video_paths.append(ep_data.cam1_temp_path)
    
    output_path = output_dir / DEFAULT_VIDEO_PATH.format(
        video_key=camera_key,
        chunk_index=chunk_idx,
        file_index=file_idx
    )
    return concatenate_videos(video_paths, output_path)


def compute_episode_stats(observations: List[np.ndarray], actions: List[np.ndarray]) -> Dict:
    """Compute statistics for an episode."""
    obs_arr = np.array(observations)
    act_arr = np.array(actions)
    
    stats = {
        'observation.state': {
            'mean': obs_arr.mean(axis=0).tolist(),
            'std': obs_arr.std(axis=0).tolist(),
            'min': obs_arr.min(axis=0).tolist(),
            'max': obs_arr.max(axis=0).tolist(),
        },
        'action': {
            'mean': act_arr.mean(axis=0).tolist(),
            'std': act_arr.std(axis=0).tolist(),
            'min': act_arr.min(axis=0).tolist(),
            'max': act_arr.max(axis=0).tolist(),
        }
    }
    return stats


def compute_global_stats(all_observations: List[np.ndarray], all_actions: List[np.ndarray]) -> Dict:
    """Compute global statistics across all frames."""
    obs_arr = np.array(all_observations)
    act_arr = np.array(all_actions)
    
    stats = {
        'observation.state': {
            'mean': obs_arr.mean(axis=0).tolist(),
            'std': obs_arr.std(axis=0).tolist(),
            'min': obs_arr.min(axis=0).tolist(),
            'max': obs_arr.max(axis=0).tolist(),
        },
        'action': {
            'mean': act_arr.mean(axis=0).tolist(),
            'std': act_arr.std(axis=0).tolist(),
            'min': act_arr.min(axis=0).tolist(),
            'max': act_arr.max(axis=0).tolist(),
        },
        'observation.images.cam_0': {
            'mean': [[[0.485]], [[0.456]], [[0.406]]],
            'std': [[[0.229]], [[0.224]], [[0.225]]],
            'min': [[[0.0]], [[0.0]], [[0.0]]],
            'max': [[[1.0]], [[1.0]], [[1.0]]],
        },
        'observation.images.cam_1': {
            'mean': [[[0.485]], [[0.456]], [[0.406]]],
            'std': [[[0.229]], [[0.224]], [[0.225]]],
            'min': [[[0.0]], [[0.0]], [[0.0]]],
            'max': [[[1.0]], [[1.0]], [[1.0]]],
        },
    }
    return stats


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_abb_to_lerobot_v30(
    raw_data_dir: Path,
    output_dir: Path,
    tasks: str,
    fps_rational: str = '50/3',
    sample_rate: int = 1,
    action_type: int = 0,
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB
):
    """Main conversion function from ABB raw data to LeRobot v3.0."""
    
    global FFMPEG_PATH
    FFMPEG_PATH = check_ffmpeg()
    if not FFMPEG_PATH:
        raise RuntimeError("FFmpeg is required but not found. Please install it first.")
    
    logging.info(f"Converting ABB raw data from {raw_data_dir} to LeRobot v3.0 at {output_dir}")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / '_temp_videos'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if (raw_data_dir / 'action.json').exists():
        episode_dirs = [raw_data_dir]
        logging.info(f"Input directory is a single episode (contains action.json)")
    else:
        episode_dirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    
    logging.info(f"Found {len(episode_dirs)} episode directories")
    
    all_episodes = []
    continuous_index = 0
    
    for episode_dir in tqdm.tqdm(episode_dirs, desc="Processing episodes"):
        ep_data = process_raw_episode(episode_dir, continuous_index, temp_dir, fps_rational, sample_rate)
        if ep_data is not None:
            all_episodes.append(ep_data)
            continuous_index += 1
    
    if len(all_episodes) == 0:
        logging.error("No valid episodes found!")
        logging.error(f"All {len(episode_dirs)} episodes were skipped. Check the error messages above.")
        return
    
    logging.info("")
    logging.info(f"{'='*60}")
    logging.info(f"Successfully processed {len(all_episodes)}/{len(episode_dirs)} episodes")
    logging.info(f"Episode indices: 0 to {len(all_episodes)-1} (continuous)")
    logging.info(f"{'='*60}")
    logging.info("")
    
    data_file_episodes = []
    cam0_video_file_episodes = []
    cam1_video_file_episodes = []
    
    current_data_episodes = []
    current_data_size = 0
    
    current_cam0_episodes = []
    current_cam0_size = 0
    
    current_cam1_episodes = []
    current_cam1_size = 0
    
    for ep_data in all_episodes:
        # ✅ 修改5: 估算大小时用6维（原来是12 = 6*2，现在仍然是6*2=12，但每个是6维不是7维）
        est_data_size = (ep_data.num_frames * 12 * 8) / (1024 * 1024)
        cam0_size = get_file_size_in_mb(ep_data.cam0_temp_path)
        cam1_size = get_file_size_in_mb(ep_data.cam1_temp_path)
        
        if current_data_size + est_data_size >= data_file_size_in_mb and len(current_data_episodes) > 0:
            data_file_episodes.append(current_data_episodes)
            current_data_episodes = []
            current_data_size = 0
        current_data_episodes.append(ep_data)
        current_data_size += est_data_size
        
        if current_cam0_size + cam0_size >= video_file_size_in_mb and len(current_cam0_episodes) > 0:
            cam0_video_file_episodes.append(current_cam0_episodes)
            current_cam0_episodes = []
            current_cam0_size = 0
        current_cam0_episodes.append(ep_data)
        current_cam0_size += cam0_size
        
        if current_cam1_size + cam1_size >= video_file_size_in_mb and len(current_cam1_episodes) > 0:
            cam1_video_file_episodes.append(current_cam1_episodes)
            current_cam1_episodes = []
            current_cam1_size = 0
        current_cam1_episodes.append(ep_data)
        current_cam1_size += cam1_size
    
    if current_data_episodes:
        data_file_episodes.append(current_data_episodes)
    if current_cam0_episodes:
        cam0_video_file_episodes.append(current_cam0_episodes)
    if current_cam1_episodes:
        cam1_video_file_episodes.append(current_cam1_episodes)
    
    logging.info(f"Will create {len(data_file_episodes)} data files, "
                 f"{len(cam0_video_file_episodes)} cam0 video files, "
                 f"{len(cam1_video_file_episodes)} cam1 video files")
    
    global_frame_offset = 0
    episode_metadata = {}
    
    for file_idx, episodes in enumerate(tqdm.tqdm(data_file_episodes, desc="Writing data files")):
        chunk_idx = file_idx // DEFAULT_CHUNK_SIZE
        file_idx_in_chunk = file_idx % DEFAULT_CHUNK_SIZE
        
        for ep_data in episodes:
            episode_metadata[ep_data.episode_index] = {
                'episode_index': ep_data.episode_index,
                'data/chunk_index': chunk_idx,
                'data/file_index': file_idx_in_chunk,
                'dataset_from_index': global_frame_offset,
                'dataset_to_index': global_frame_offset + ep_data.num_frames,
            }
            global_frame_offset += ep_data.num_frames
        
        start_offset = episode_metadata[episodes[0].episode_index]['dataset_from_index']
        write_combined_data_file(episodes, output_dir, chunk_idx, file_idx_in_chunk, start_offset)
    
    cam0_key = 'observation.images.cam_0'
    for file_idx, episodes in enumerate(tqdm.tqdm(cam0_video_file_episodes, desc="Writing cam0 videos")):
        chunk_idx = file_idx // DEFAULT_CHUNK_SIZE
        file_idx_in_chunk = file_idx % DEFAULT_CHUNK_SIZE
        
        if not write_combined_video_file(episodes, output_dir, cam0_key, chunk_idx, file_idx_in_chunk):
            logging.error(f"Failed to write cam0 video file {chunk_idx}/{file_idx_in_chunk}")
        
        cumulative_duration = 0.0
        for ep_data in episodes:
            ep_duration = ep_data.num_frames / ep_data.fps
            episode_metadata[ep_data.episode_index].update({
                f'videos/{cam0_key}/chunk_index': chunk_idx,
                f'videos/{cam0_key}/file_index': file_idx_in_chunk,
                f'videos/{cam0_key}/from_timestamp': cumulative_duration,
                f'videos/{cam0_key}/to_timestamp': cumulative_duration + ep_duration,
            })
            cumulative_duration += ep_duration
    
    cam1_key = 'observation.images.cam_1'
    for file_idx, episodes in enumerate(tqdm.tqdm(cam1_video_file_episodes, desc="Writing cam1 videos")):
        chunk_idx = file_idx // DEFAULT_CHUNK_SIZE
        file_idx_in_chunk = file_idx % DEFAULT_CHUNK_SIZE
        
        if not write_combined_video_file(episodes, output_dir, cam1_key, chunk_idx, file_idx_in_chunk):
            logging.error(f"Failed to write cam1 video file {chunk_idx}/{file_idx_in_chunk}")
        
        cumulative_duration = 0.0
        for ep_data in episodes:
            ep_duration = ep_data.num_frames / ep_data.fps
            episode_metadata[ep_data.episode_index].update({
                f'videos/{cam1_key}/chunk_index': chunk_idx,
                f'videos/{cam1_key}/file_index': file_idx_in_chunk,
                f'videos/{cam1_key}/from_timestamp': cumulative_duration,
                f'videos/{cam1_key}/to_timestamp': cumulative_duration + ep_duration,
            })
            cumulative_duration += ep_duration
    
    shutil.rmtree(temp_dir)
    
    all_observations = []
    all_actions = []
    for ep_data in all_episodes:
        all_observations.extend(ep_data.observations)
        all_actions.extend(ep_data.actions)
    
    global_stats = compute_global_stats(all_observations, all_actions)
    
    all_episode_stats = []
    for ep_data in all_episodes:
        ep_stats = compute_episode_stats(ep_data.observations, ep_data.actions)
        all_episode_stats.append(ep_stats)
    
    episodes_records = []
    for idx, ep_data in enumerate(all_episodes):
        ep_idx = ep_data.episode_index
        record = {
            **episode_metadata[ep_idx],
            'tasks': [tasks],
            'length': ep_data.num_frames,
            'meta/episodes/chunk_index': 0,
            'meta/episodes/file_index': 0,
            **flatten_dict({'stats': all_episode_stats[idx]})
        }
        episodes_records.append(record)
    
    df_episodes = pd.DataFrame(episodes_records)
    episodes_path = output_dir / 'meta' / 'episodes' / 'chunk-000' / 'file_000.parquet'
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    df_episodes.to_parquet(episodes_path, index=False)
    logging.info(f"Written episodes metadata to {episodes_path}")
    
    stats_json_path = output_dir / 'meta' / 'stats.json'
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_json_path, 'w') as f:
        json.dump(global_stats, f, indent=2)
    logging.info(f"Written stats.json to {stats_json_path}")
    
    df_stats = pd.DataFrame([flatten_dict(global_stats)])
    stats_parquet_path = output_dir / 'meta' / 'stats.parquet'
    df_stats.to_parquet(stats_parquet_path, index=False)
    logging.info(f"Written stats.parquet to {stats_parquet_path}")
    
    df_tasks = pd.DataFrame(
        {'task_index': [0]},
        index=[tasks]
    )
    
    tasks_path = output_dir / 'meta' / 'tasks.parquet'
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    df_tasks.to_parquet(tasks_path)
    logging.info(f"Written tasks metadata to {tasks_path}")
    
    first_video_path = output_dir / DEFAULT_VIDEO_PATH.format(
        video_key='observation.images.cam_0', chunk_index=0, file_index=0)
    video_info = get_video_info(first_video_path)
    
    total_frames = sum(ep.num_frames for ep in all_episodes)
    
    if action_type == 0:
        n_shape = 7
        action_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
    elif action_type == 1:
        n_shape = 6
        action_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    # ✅ 修改6: info.json中的features改为6维
    info = {
        'codebase_version': CODEBASE_VERSION,
        'robot_type': 'abb_robot',
        'total_episodes': len(all_episodes),
        'total_frames': total_frames,
        'total_tasks': 1,
        'fps': int(video_info['fps']),
        'data_files_size_in_mb': data_file_size_in_mb,
        'video_files_size_in_mb': video_file_size_in_mb,
        'splits': {
            'train': f'0:{len(all_episodes)}'
        },
        'data_path': DEFAULT_DATA_PATH,
        'video_path': DEFAULT_VIDEO_PATH,
        'features': {
            'observation.state': {
                'dtype': 'float32',
                'shape': [n_shape],  # ✅ 改为6
                'names': action_names,  # ✅ 去掉gripper
            },
            'action': {
                'dtype': 'float32',
                'shape': [n_shape],  # ✅ 改为6
                'names': action_names,  # ✅ 去掉gripper
            },
            'observation.images.cam_0': {
                'dtype': 'video',
                'shape': [video_info['height'], video_info['width'], 3],
                'names': ['height', 'width', 'channels'],
                'info': {
                    'video.fps': int(video_info['fps']),
                    'video.height': video_info['height'],
                    'video.width': video_info['width'],
                    'video.channels': 3,
                    'video.codec': 'h264',
                    'video.pix_fmt': 'yuv420p',
                    'video.is_depth_map': False,
                    'has_audio': False
                }
            },
            'observation.images.cam_1': {
                'dtype': 'video',
                'shape': [video_info['height'], video_info['width'], 3],
                'names': ['height', 'width', 'channels'],
                'info': {
                    'video.fps': int(video_info['fps']),
                    'video.height': video_info['height'],
                    'video.width': video_info['width'],
                    'video.channels': 3,
                    'video.codec': 'h264',
                    'video.pix_fmt': 'yuv420p',
                    'video.is_depth_map': False,
                    'has_audio': False
                }
            },
            'timestamp': {
                'dtype': 'float32',
                'shape': [1],
                'names': None,
            },
            'frame_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'episode_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            },
            'task_index': {
                'dtype': 'int64',
                'shape': [1],
                'names': None,
            }
        }
    }
    
    info_path = output_dir / 'meta' / 'info.json'
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logging.info(f"Written info.json to {info_path}")
    
    logging.info("")
    logging.info(f"{'='*60}")
    logging.info(f"✓ CONVERSION COMPLETED SUCCESSFULLY")
    logging.info(f"{'='*60}")
    logging.info(f"  Dataset version: {CODEBASE_VERSION}")
    logging.info(f"  Episodes: {len(all_episodes)} (indices 0-{len(all_episodes)-1})")
    logging.info(f"  Total frames: {total_frames}")
    logging.info(f"  FPS: {int(video_info['fps'])}")
    logging.info(f"  Resolution: {video_info['width']}x{video_info['height']}")
    logging.info(f"  State dim: 6 (joint only, no gripper)")  # ✅ 更新日志
    logging.info(f"  Action dim: 6 (joint only, no gripper)")  # ✅ 更新日志
    logging.info(f"  Output directory: {output_dir}")
    logging.info(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert ABB raw data directly to LeRobot Dataset v3.0 format'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing raw ABB episode folders'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for LeRobot v3.0 dataset'
    )
    parser.add_argument(
        '--fps',
        type=str,
        default='25',
        help='Target FPS as rational number (default: 25 fps)'
    )
    parser.add_argument(
        '--data-file-size',
        type=int,
        default=DEFAULT_DATA_FILE_SIZE_IN_MB,
        help=f'Maximum data file size in MB (default: {DEFAULT_DATA_FILE_SIZE_IN_MB})'
    )
    parser.add_argument(
        '--video-file-size',
        type=int,
        default=DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        help=f'Maximum video file size in MB (default: {DEFAULT_VIDEO_FILE_SIZE_IN_MB})'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default=r"Pick up the box on the conveyor belt and place it into the blue plastic bin."
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        help = f"downsample rates: 2, 4, 8, 16"
    )

    parser.add_argument(
        "--action-type",
        type=int,
        help="0: joints+gripper, 1: joints 6"
    )

    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    sample_rate = args.sample_rate
    
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        exit(1)
    
    convert_abb_to_lerobot_v30(
        raw_data_dir=input_dir,
        output_dir=output_dir,
        fps_rational=args.fps,
        data_file_size_in_mb=args.data_file_size,
        video_file_size_in_mb=args.video_file_size,
        tasks=args.task,
        sample_rate=sample_rate,
        action_type=args.action_type
    )