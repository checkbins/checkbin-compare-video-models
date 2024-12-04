from modal import Image as ModalImage, App, gpu, Secret, Mount, enter, build, method

checkbin_app_key = "compare_video_models"
test_prompts_path = "prompts.json"

run_ltx_video_image = (
    ModalImage.from_registry(
       "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install([
        "git",
        "ffmpeg",
    ])
    .run_commands([
        "git clone https://github.com/Lightricks/LTX-Video.git /root/LTX-Video",
    ])
    .pip_install(
        "argparse",
        "diffusers==0.30.2",
        "imageio==2.34.0",
        "numpy==1.24.4",
        "safetensors==0.4.3",
        "torch==2.1.1",
        "torchvision==0.16.1",
        "Pillow",
        "transformers==4.39.3",
        "opencv-python==4.9.0.80",
        "boto3",
        "tinydb",
        "google-cloud-storage",
        "azure-storage-blob",
        "azure-datalake-store",
        "azure-storage-file-datalake",
        "einops==0.7.0",
        "sentencepiece",
        'imageio[ffmpeg]',
        'beautifulsoup4',
        
    )
    .apt_install([
        "libgl1-mesa-dev",
        "libglib2.0-0",
    ])
)

app = App("compare-video-models-ltx", image=run_ltx_video_image)

def run_inference(checkbins):
    print("Running inference")

@app.cls(
    gpu=gpu.A100(size="80GB"),
    timeout=86400,
    image=run_ltx_video_image,
    secrets=[Secret.from_name("checkbin-secret"), Secret.from_name("huggingface-secret")],
    mounts=[Mount.from_local_dir("./checkbin-python", remote_path="/root/checkbin-python"),
            Mount.from_local_dir("./inputs", remote_path="/root/inputs")]
)
class Model: 
    @build()
    def setup(self):
        from huggingface_hub import snapshot_download
        snapshot_download("Lightricks/LTX-Video", local_dir="/root/LTX-Video/ckpts", local_dir_use_symlinks=False, repo_type='model')

    @method()
    def run_inference(self):
        print("Running inference")

        import os, sys

        sys.path.insert(0, 'checkbin-python/src')
        import checkbin
        checkbin.authenticate(token=os.environ["CHECKBIN_TOKEN"])
        checkbin_app = checkbin.App(app_key=checkbin_app_key, mode="remote")

        ckpt_folders = [f for f in os.listdir('/root/LTX-Video/ckpts') if os.path.isdir(os.path.join('/root/LTX-Video/ckpts', f))]
        print("Checkpoint folders:", ckpt_folders)

        with checkbin_app.start_run(set_id="6256aa4d-b84f-4c84-9778-941f7c3816f3") as bin_generator:
            for checkbin in bin_generator:
                negative_prompt = checkbin.get_input_data('negative_prompt')
                prompt = checkbin.get_input_data('prompt')
                print(f"Running inference for prompt: {prompt}")
                os.system(
                    f"cd /root/LTX-Video && "
                    f"python inference.py "
                    f"--ckpt_dir '/root/LTX-Video/ckpts' "
                    f"--prompt '{prompt}' "
                    f"--height 720 "
                    f"--width 1280 "
                    f"--num_frames 129 "
                    f"--seed 42"
                )
                import os

                checkbin.checkin("LTX Video")
                
                for root, dirs, files in os.walk('/root/LTX-Video/outputs'):
                    level = root.replace('/root/LTX-Video/outputs', '').count(os.sep)
                    indent = ' ' * 4 * (level)
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for f in files:
                        print(f"{subindent}{f}")

                # Delete all mp4 files in the outputs directory
                mp4_files = []
                for root, dirs, files in os.walk('/root/LTX-Video/outputs'):
                    for f in files:
                        if f.endswith('.mp4'):
                            mp4_files.append(os.path.join(root, f))
                for mp4_file in mp4_files:
                    print(mp4_file)
                    print("found a file!")
                    try:
                        # Add it to Checkbin (!) 
                        path = os.path.join('/root/LTX-Video/outputs', mp4_file)
                        print(f"Found video at {path}!!")
                        checkbin.upload_file("inference_output", str(path), "video")
                        os.remove(path)
                        print(f"Deleted file: {mp4_file}")
                    except Exception as e:
                        print(f"Error deleting file {mp4_file}: {e}")

                checkbin.submit()