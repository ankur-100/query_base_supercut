import subprocess
import os
from typing import List
import tempfile
from get_chunks import get_relevant_chunks, Chunk

def create_video_segments(video_url: str, chunks: List[Chunk], output_path: str = "output.mp4"):

    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        
        print(f"Processing {len(chunks)} segments...")
        
        for i, chunk in enumerate(chunks):
            segment_file = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            segment_files.append(segment_file)
            
            print(f"Extracting segment {i+1}/{len(chunks)}: {chunk.start:.1f}s - {chunk.start + chunk.duration:.1f}s")
            
            # cmd = [
            #     "ffmpeg",
            #     "-ss", str(chunk.start),  # Start time
            #     "-i", f"$(yt-dlp -g -f 'best[ext=mp4]' '{video_url}')",  # Get direct URL from yt-dlp
            #     "-t", str(chunk.duration),  # Duration
            #     "-c", "copy",  # Copy codec (no re-encoding for speed)
            #     "-avoid_negative_ts", "make_zero",
            #     "-y",  # Overwrite output
            #     segment_file
            # ]
            
            shell_cmd = f"""ffmpeg -ss {chunk.start} -i "$(yt-dlp -g -f 'best[ext=mp4]' '{video_url}')" -t {chunk.duration} -c copy -avoid_negative_ts make_zero -y "{segment_file}" """
            
            try:
                subprocess.run(shell_cmd, shell=True, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error extracting segment {i+1}: {e.stderr}")
                raise
        
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for segment_file in segment_files:
                f.write(f"file '{segment_file}'\n")
        
        print("Stitching segments together...")
        
        concat_cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-y",
            output_path
        ]
        
        try:
            subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Video created successfully: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating segments: {e.stderr}")
            raise


def stitch(url_or_id: str, query: str):
    
    print(f"Finding relevant chunks for query: '{query}'")
    chunks = get_relevant_chunks(url_or_id, query)
    
    if not chunks:
        print("No relevant chunks found!")
        return
    
    print(f"\nFound {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. [{chunk.start:.1f}s - {chunk.start + chunk.duration:.1f}s]: {chunk.text[:50]}...")
    
    output_path = "stitched_video.mp4"
    create_video_segments(url_or_id, chunks, output_path)
    total_duration = sum(chunk.duration for chunk in chunks)
    print(f"\nTotal duration of stitched video: {total_duration:.1f} seconds")


if __name__ == "__main__":    
    stitch(
        url_or_id="https://www.youtube.com/watch?v=XMGvGvp2a6M", 
        query="negatives of the movie"
    )