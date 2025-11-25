#!/usr/bin/env python
"""
Direct conversion from ABB raw data to LeRobot Dataset v3.0 format.
Combines logic from abb->v2.0 and v2.1->v3.0 conversion scripts.

Fixed version:
1. Correct stats.json format
2. Continuous episode indices (re-indexed after dropping invalid episodes)
3. Both action and state are 6D joint values
"""

import os
import json
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

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


# if dataset_type: joint6
OBS_NAMES=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
ACTION_NAMES=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper']
# dataset_type: full14
OBS_NAMES=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',  'gripper']

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


def parse_action_states(action_json_path: Path) -> List[Dict]:
    """Parse states list from action.json."""
    with open(action_json_path, 'r') as f:
        data = json.load(f)
    states = data.get('states', [])
    result: List[Dict] = []
    
    for st in states:
        if not isinstance(st, dict):
            continue
        joint = st.get('joint')
        gripper = st.get('gripper')
        
        if joint is None:
            continue
        if not isinstance(joint, list) or len(joint) != 6:
            continue
        
        # Parse gripper: "on" -> 1.0, "off" -> 0.0
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
        })
    return result


class EpisodeData:
    """Container for episode data."""
    def __init__(self, episode_index: int, episode_dir: Path):
        self.episode_index = episode_index  # This will be the NEW continuous index
        self.episode_dir = episode_dir
        self.states_data = []
        self.cam0_temp_path = None
        self.cam1_temp_path = None
        self.num_frames = 0
        self.fps = 0.0
        self.observations = []  # 6D joint state
        self.actions = []       # 6D joint action (same as state, or shifted)
        self.timestamps = []
        

def process_raw_episode(episode_dir: Path, episode_index: int, temp_dir: Path, fps_rational: str) -> EpisodeData:
    """Process a single raw episode directory.
    
    Args:
        episode_dir: Path to the raw episode directory
        episode_index: The NEW continuous index for this episode
        temp_dir: Temporary directory for re-encoded videos
        fps_rational: Target FPS as rational number
    """
    ep_data = EpisodeData(episode_index, episode_dir)
    
    # Find video files
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
    
    # Parse action states
    try:
        states_data = parse_action_states(action_json)
    except Exception as e:
        logging.warning(f"Skipped {episode_dir.name}: failed to parse action.json - {e}")
        return None
        
    if len(states_data) == 0:
        logging.warning(f"Skipped {episode_dir.name}: no valid states in action.json")
        return None
    
    # Extract joints and grippers
    joints = [state['joint'] for state in states_data]
    grippers = [state['gripper'] for state in states_data]
    
    # Log gripper distribution for debugging
    on_count = sum(1 for g in grippers if g == 1.0)
    off_count = sum(1 for g in grippers if g == 0.0)
    logging.debug(f"{episode_dir.name}: gripper on={on_count}, off={off_count}")
    
    # Re-encode videos to temp directory
    ep_data.cam0_temp_path = temp_dir / f'cam0_ep{episode_index:06d}.mp4'
    ep_data.cam1_temp_path = temp_dir / f'cam1_ep{episode_index:06d}.mp4'
    
    if not run_ffmpeg_reencode(cam0_in, ep_data.cam0_temp_path, fps_rational):
        logging.warning(f"Skipped {episode_dir.name}: failed to re-encode cam0 video")
        return None
    if not run_ffmpeg_reencode(cam1_in, ep_data.cam1_temp_path, fps_rational):
        logging.warning(f"Skipped {episode_dir.name}: failed to re-encode cam1 video")
        return None
    
    # Get video info
    try:
        info0 = get_video_info(ep_data.cam0_temp_path)
        info1 = get_video_info(ep_data.cam1_temp_path)
    except Exception as e:
        logging.warning(f"Skipped {episode_dir.name}: failed to get video info - {e}")
        return None
    
    # Align frames (skip first frame as in original script)
    available_frames = min(info0.get('nb_frames', 0), info1.get('nb_frames', 0))
    max_aligned = min(len(joints), available_frames)
    if max_aligned <= 1:
        logging.warning(f"Skipped {episode_dir.name}: insufficient frames after alignment (only {max_aligned})")
        return None
    
    # Skip first frame and align
    use_num = max_aligned - 1
    joints = joints[:max_aligned][1:]
    grippers = grippers[:max_aligned][1:]
    
    ep_data.fps = float(info0.get('fps', 0.0)) or (50.0/3.0)
    # (o_0, a_1), so skip a_0
    ep_data.num_frames = use_num - 1
    
    # Generate timestamps
    timestamps = (np.arange(use_num, dtype=np.float32) / np.float32(ep_data.fps))
    
    # Build observations and actions (both are 7D: joint + gripper)
    for frame_idx in range(use_num - 1):
        # 7D state: 6 joint angles + 1 gripper
        state7 = np.concatenate([joints[frame_idx], [grippers[frame_idx]]], dtype=np.float32)
        # 7D action: same as state
        action7 = np.concatenate([joints[frame_idx+1], [grippers[frame_idx+1]]], dtype=np.float32)
        
        ep_data.observations.append(state7)
        ep_data.actions.append(action7)
        ep_data.timestamps.append(float(timestamps[frame_idx]))
    
    logging.info(f"✓ Processed {episode_dir.name} -> episode_{episode_index}: {ep_data.num_frames} frames @ {ep_data.fps:.2f} fps")
    return ep_data


def write_combined_data_file(episodes: List[EpisodeData], output_dir: Path, chunk_idx: int, file_idx: int, global_frame_offset: int):
    """Write combined data file for multiple episodes."""
    all_records = []
    
    # jsonl_path = "episode0.jsonl"   # 你想写入的路径
    
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
            # if idx == 0:
            #     with open(jsonl_path, "a") as f1:
            #         f1.write(json.dumps(record) + "\n")
            all_records.append(record)
        global_frame_offset += ep_data.num_frames
    
    df = pd.DataFrame(all_records)
    path = output_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    
    return global_frame_offset


def write_combined_video_file(episodes: List[EpisodeData], output_dir: Path, camera_key: str, chunk_idx: int, file_idx: int):
    """Write combined video file for multiple episodes.
    
    camera_key should be the full feature key like 'observation.images.cam_0'
    """
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
        # Image features need stats entries too (using standard image normalization values)
        'observation.images.cam_0': {
            'mean': [[[0.485]], [[0.456]], [[0.406]]],  # ImageNet mean (C, 1, 1)
            'std': [[[0.229]], [[0.224]], [[0.225]]],   # ImageNet std (C, 1, 1)
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
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB
):
    """Main conversion function from ABB raw data to LeRobot v3.0."""
    
    global FFMPEG_PATH
    FFMPEG_PATH = check_ffmpeg()
    if not FFMPEG_PATH:
        raise RuntimeError("FFmpeg is required but not found. Please install it first.")
    
    logging.info(f"Converting ABB raw data from {raw_data_dir} to LeRobot v3.0 at {output_dir}")
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory for re-encoded videos
    temp_dir = output_dir / '_temp_videos'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all episode directories
    # Support two modes:
    # 1. If input dir itself is an episode (contains action.json), process it as single episode
    # 2. If input dir contains subdirectories, treat each subdir as an episode
    
    if (raw_data_dir / 'action.json').exists():
        # Mode 1: Input is a single episode directory
        episode_dirs = [raw_data_dir]
        logging.info(f"Input directory is a single episode (contains action.json)")
    else:
        # Mode 2: Input contains multiple episode subdirectories
        episode_dirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    
    logging.info(f"Found {len(episode_dirs)} episode directories")
    
    # Process all episodes with CONTINUOUS indexing
    all_episodes = []
    continuous_index = 0  # This will be the new continuous episode index
    
    for episode_dir in tqdm.tqdm(episode_dirs, desc="Processing episodes"):
        # Pass the continuous_index as the episode_index
        ep_data = process_raw_episode(episode_dir, continuous_index, temp_dir, fps_rational)
        if ep_data is not None:
            all_episodes.append(ep_data)
            continuous_index += 1  # Only increment when episode is successfully processed
    
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
    
    # Organize episodes into data/video files based on size limits
    data_file_episodes = []
    cam0_video_file_episodes = []
    cam1_video_file_episodes = []
    
    current_data_episodes = []
    current_data_size = 0
    
    current_cam0_episodes = []
    current_cam0_size = 0
    
    current_cam1_episodes = []
    current_cam1_size = 0
    
    # Estimate sizes
    for ep_data in all_episodes:
        # Estimate data size (rough: 8 bytes per float * 12 features * num_frames / 1MB)
        est_data_size = (ep_data.num_frames * 12 * 8) / (1024 * 1024)
        cam0_size = get_file_size_in_mb(ep_data.cam0_temp_path)
        cam1_size = get_file_size_in_mb(ep_data.cam1_temp_path)
        
        # Group for data files
        if current_data_size + est_data_size >= data_file_size_in_mb and len(current_data_episodes) > 0:
            data_file_episodes.append(current_data_episodes)
            current_data_episodes = []
            current_data_size = 0
        current_data_episodes.append(ep_data)
        current_data_size += est_data_size
        
        # Group for cam0 video files
        if current_cam0_size + cam0_size >= video_file_size_in_mb and len(current_cam0_episodes) > 0:
            cam0_video_file_episodes.append(current_cam0_episodes)
            current_cam0_episodes = []
            current_cam0_size = 0
        current_cam0_episodes.append(ep_data)
        current_cam0_size += cam0_size
        
        # Group for cam1 video files
        if current_cam1_size + cam1_size >= video_file_size_in_mb and len(current_cam1_episodes) > 0:
            cam1_video_file_episodes.append(current_cam1_episodes)
            current_cam1_episodes = []
            current_cam1_size = 0
        current_cam1_episodes.append(ep_data)
        current_cam1_size += cam1_size
    
    # Add remaining episodes
    if current_data_episodes:
        data_file_episodes.append(current_data_episodes)
    if current_cam0_episodes:
        cam0_video_file_episodes.append(current_cam0_episodes)
    if current_cam1_episodes:
        cam1_video_file_episodes.append(current_cam1_episodes)
    
    logging.info(f"Will create {len(data_file_episodes)} data files, "
                 f"{len(cam0_video_file_episodes)} cam0 video files, "
                 f"{len(cam1_video_file_episodes)} cam1 video files")
    
    # Write data files
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
        
        # Reset global_frame_offset for write function
        start_offset = episode_metadata[episodes[0].episode_index]['dataset_from_index']
        write_combined_data_file(episodes, output_dir, chunk_idx, file_idx_in_chunk, start_offset)
    
    # Write video files for cam0
    cam0_key = 'observation.images.cam_0'
    for file_idx, episodes in enumerate(tqdm.tqdm(cam0_video_file_episodes, desc="Writing cam0 videos")):
        chunk_idx = file_idx // DEFAULT_CHUNK_SIZE
        file_idx_in_chunk = file_idx % DEFAULT_CHUNK_SIZE
        
        if not write_combined_video_file(episodes, output_dir, cam0_key, chunk_idx, file_idx_in_chunk):
            logging.error(f"Failed to write cam0 video file {chunk_idx}/{file_idx_in_chunk}")
        
        # Calculate timestamps for each episode in this video file
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
    
    # Write video files for cam1
    cam1_key = 'observation.images.cam_1'
    for file_idx, episodes in enumerate(tqdm.tqdm(cam1_video_file_episodes, desc="Writing cam1 videos")):
        chunk_idx = file_idx // DEFAULT_CHUNK_SIZE
        file_idx_in_chunk = file_idx % DEFAULT_CHUNK_SIZE
        
        if not write_combined_video_file(episodes, output_dir, cam1_key, chunk_idx, file_idx_in_chunk):
            logging.error(f"Failed to write cam1 video file {chunk_idx}/{file_idx_in_chunk}")
        
        # Calculate timestamps for each episode in this video file
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
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Collect all observations and actions for global stats
    all_observations = []
    all_actions = []
    for ep_data in all_episodes:
        all_observations.extend(ep_data.observations)
        all_actions.extend(ep_data.actions)
    
    # Compute global statistics
    global_stats = compute_global_stats(all_observations, all_actions)
    
    # Compute per-episode statistics
    all_episode_stats = []
    for ep_data in all_episodes:
        ep_stats = compute_episode_stats(ep_data.observations, ep_data.actions)
        all_episode_stats.append(ep_stats)
    
    # Write episodes metadata
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
    
    # Write stats.json (correct format)
    stats_json_path = output_dir / 'meta' / 'stats.json'
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_json_path, 'w') as f:
        json.dump(global_stats, f, indent=2)
    logging.info(f"Written stats.json to {stats_json_path}")
    
    # Also write stats.parquet for compatibility
    df_stats = pd.DataFrame([flatten_dict(global_stats)])
    stats_parquet_path = output_dir / 'meta' / 'stats.parquet'
    df_stats.to_parquet(stats_parquet_path, index=False)
    logging.info(f"Written stats.parquet to {stats_parquet_path}")
    
    # Write tasks metadata
    df_tasks = pd.DataFrame(
        {'task_index': [0]},
        index=[tasks]
    )
    tasks_path = output_dir / 'meta' / 'tasks.parquet'
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    df_tasks.to_parquet(tasks_path, index=False)
    logging.info(f"Written tasks metadata to {tasks_path}")
    
    # Get video info from first episode
    first_video_path = output_dir / DEFAULT_VIDEO_PATH.format(
        video_key='observation.images.cam_0', chunk_index=0, file_index=0)
    video_info = get_video_info(first_video_path)
    
    # Write info.json
    total_frames = sum(ep.num_frames for ep in all_episodes)
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
                'shape': [7],
                'names': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper'],
            },
            'action': {
                'dtype': 'float32',
                'shape': [7],
                'names': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper'],
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
    logging.info(f"  State dim: 7 (joint + gripper)")
    logging.info(f"  Action dim: 7 (joint + gripper)")
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
        default='50/3',
        help='Target FPS as rational number (default: 50/3 ≈ 16.67 fps)'
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
        default=r"Pick up blocks from conveyor belt and place them in boxes"
    )

    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        exit(1)
    
    convert_abb_to_lerobot_v30(
        raw_data_dir=input_dir,
        output_dir=output_dir,
        fps_rational=args.fps,
        data_file_size_in_mb=args.data_file_size,
        video_file_size_in_mb=args.video_file_size,
        tasks=args.task
    )