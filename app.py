# # app.py
# import streamlit as st
# import os
# from pipeline import VideoDetectionPipeline
# import asyncio
# import sys
# import torch 
# import types

# if sys.platform == "win32":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# # Create a dummy __path__ that won't crash
# if isinstance(torch.classes, types.ModuleType):
#     torch.classes.__path__ = []

# st.set_page_config(page_title="YOLO AVI Detection", layout="centered")
# st.title("🎥 YOLOv8 AVI Batch Detection Pipeline")

# if "processed_videos" not in st.session_state:
#     st.session_state.processed_videos = {}

# uploaded_files = st.file_uploader("Upload one or more .avi files", type=["avi"], accept_multiple_files=True)

# if uploaded_files:
#     model_path = "best_v8_4.pt"
#     pipeline = VideoDetectionPipeline(model_path=model_path)
#     os.makedirs("uploads", exist_ok=True)

#     for uploaded_file in uploaded_files:
#         video_name = uploaded_file.name
#         save_path = os.path.join("uploads", video_name)

#         # Save file if not already saved
#         if not os.path.exists(save_path):
#             with open(save_path, "wb") as f:
#                 f.write(uploaded_file.read())

#         st.markdown(f"---\n### 📁 Processing `{video_name}`")

#         # Check if already processed
#         if video_name in st.session_state.processed_videos:
#             zip_path = st.session_state.processed_videos[video_name]
#             st.success("✅ Already processed.")
#         else:
#             progress_bar = st.progress(0, text=f"Starting detection for {video_name}...")

#             def update_progress(p):
#                 percent = int(p * 100)
#                 progress_bar.progress(percent, text=f"{percent}% done for {video_name}")

#             # Run pipeline with progress callback
#             with st.spinner(f"🔍 Running detection on `{video_name}`..."):
#                 zip_path = pipeline.run(
#                     video_path=save_path,
#                     output_base_dir="output",
#                     progress_callback=update_progress
#                 )
#                 st.success(f"✅ Detection complete: `{video_name}`")
#                 st.session_state.processed_videos[video_name] = zip_path

#         # Show download
#         with open(zip_path, "rb") as zip_file:
#             st.download_button(
#                 label=f"📦 Download Results for {video_name}",
#                 data=zip_file,
#                 file_name=os.path.basename(zip_path),
#                 mime="application/zip"
#             )


# app.py
import streamlit as st
import os
from pipeline import VideoDetectionPipeline
import asyncio
import sys
import torch
import types

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

st.set_page_config(page_title="YOLO AVI Batch Detection", layout="centered")
st.title("🎥 YOLOv8 AVI Batch Detection from Folder")

if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = {}

root_dir = st.text_input("Enter the full path to the root folder containing subfolders with .avi videos:")

if st.button("Start Batch Processing") and root_dir:
    model_path = "best_v8_4.pt"
    pipeline = VideoDetectionPipeline(model_path=model_path)

    if not os.path.exists(root_dir):
        st.error("❌ Provided root directory does not exist.")
    else:
        subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]

        for subfolder in subfolders:
            video_files = [f for f in os.listdir(subfolder) if f.endswith(".avi")]
            if not video_files:
                continue  # Skip if no .avi video

            video_path = os.path.join(subfolder, video_files[0])
            video_name = os.path.basename(video_path)
            st.markdown(f"---\n### 📁 Processing `{video_name}`")

            # Skip if already processed
            if video_name in st.session_state.processed_videos:
                zip_path = st.session_state.processed_videos[video_name]
                st.success("✅ Already processed.")
            else:
                progress_bar = st.progress(0, text=f"Starting detection for {video_name}...")

                def update_progress(p):
                    percent = int(p * 100)
                    progress_bar.progress(percent, text=f"{percent}% done for {video_name}")

                with st.spinner(f"🔍 Running detection on `{video_name}`..."):
                    zip_path = pipeline.run(
                        video_path=video_path,
                        output_base_dir="output_20240826",
                        progress_callback=update_progress
                    )
                    st.success(f"✅ Detection complete: `{video_name}`")
                    st.session_state.processed_videos[video_name] = zip_path

            # with open(zip_path, "rb") as zip_file:
            #     st.download_button(
            #         label=f"📦 Download Results for {video_name}",
            #         data=zip_file,
            #         file_name=os.path.basename(zip_path),
            #         mime="application/zip"
            #     )
