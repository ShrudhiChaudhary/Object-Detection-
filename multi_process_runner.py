import multiprocessing
from pipeline import VideoDetectionPipeline

def process_single_video(args):
    video_path, output_dir, model_path = args
    pipeline = VideoDetectionPipeline(model_path)
    try:
        return pipeline.run(video_path, output_dir)
    except Exception as e:
        return f"Failed: {video_path} -> {e}"

def run_batch_in_parallel(video_paths, output_dir, model_path):
    args_list = [(vp, output_dir, model_path) for vp in video_paths]
    with multiprocessing.Pool(processes=min(4, len(video_paths))) as pool:
        results = pool.map(process_single_video, args_list)
    return results
