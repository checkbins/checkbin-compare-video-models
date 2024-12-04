from modal import Image as ModalImage, App, build, enter, build, method, gpu, Secret, Mount

checkbin_app_key = "compare_video_models"
test_prompts_path = "prompts_01.json"

run_hunyuan_video_model = (
    ModalImage.from_registry(
        "hunyuanvideo/hunyuanvideo:cuda_12"
    )
    .run_commands([
        "git clone https://github.com/Tencent/HunyuanVideo /root/HunyuanVideo",
    ])
    .run_commands([
        "conda install conda=24.11.0",
    ])
    .apt_install([
        "libgl1-mesa-dev",
        "libglib2.0-0",
    ])
    .pip_install([
        "torch==2.1.1",
        "torchvision==0.16.1",
        "opencv-python==4.9.0.80",
        "diffusers==0.30.2",
        "transformers==4.39.3",
        "tokenizers==0.15.2",
        "accelerate==1.1.1",
        "pandas==2.0.3",
        "numpy==1.24.4",
        "einops==0.7.0",
        "tqdm==4.66.2",
        "loguru==0.7.2",
        "imageio==2.34.0",
        "imageio-ffmpeg==0.5.1",
        "safetensors==0.4.3",
        "opencv-python==4.9.0.80",
        "boto3",
        "tinydb",
        "google-cloud-storage",
        "azure-storage-blob",
        "azure-datalake-store",
        "azure-storage-file-datalake",
    ])
    .apt_install([
        "git",
        "libglib2.0-0",
    ])
    .run_commands([
        "pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.9.post1"
    ])
)

app = App("compare-video-models", image=run_hunyuan_video_model)

@app.cls(
    gpu=gpu.A100(size="80GB"),
    timeout=86400,
    image=run_hunyuan_video_model,
    secrets=[Secret.from_name("checkbin-secret"), Secret.from_name("huggingface-secret")],
    mounts=[Mount.from_local_dir("./checkbin-python", remote_path="/root/checkbin-python"),
            Mount.from_local_dir("./inputs", remote_path="/root/inputs")]
)
class Model: 
    @build()
    def setup(self):
        import os
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="tencent/HunyuanVideo", local_dir="/root/HunyuanVideo/ckpts")
        snapshot_download(repo_id="xtuner/llava-llama-3-8b-v1_1-transformers", local_dir="/root/HunyuanVideo/ckpts/llava-llama-3-8b-v1_1-transformers")
        os.system(
            "cd /root/HunyuanVideo && "
            "python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py "
            "--input_dir ckpts/llava-llama-3-8b-v1_1-transformers "
            "--output_dir ckpts/text_encoder"
        )
        snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir="/root/HunyuanVideo/ckpts/text_encoder_2")

    @method()
    def run_inference(self):
        import os, sys

        sys.path.insert(0, 'checkbin-python/src')
        import checkbin
        checkbin.authenticate(token=os.environ["CHECKBIN_TOKEN"])
        checkbin_app = checkbin.App(app_key=checkbin_app_key, mode="remote")

        ckpt_folders = [f for f in os.listdir('/root/HunyuanVideo/ckpts') if os.path.isdir(os.path.join('/root/HunyuanVideo/ckpts', f))]
        print("Checkpoint folders:", ckpt_folders)


        with checkbin_app.start_run(set_id="6256aa4d-b84f-4c84-9778-941f7c3816f3") as bin_generator:
            for checkbin in bin_generator:
                negative_prompt = checkbin.get_input_data('negative_prompt')
                prompt = checkbin.get_input_data('prompt')
                print(f"Running inference for prompt: {prompt}")
                os.system(
                    "cd /root/HunyuanVideo && "
                    "python sample_video.py "
                    "--video-size 720 1280 "
                    "--video-length 129 "
                    "--infer-steps 50 "
                    f"--prompt '{prompt}' "
                    "--flow-reverse "
                    "--use-cpu-offload "
                    "--save-path ./results"
                )
                checkbin.checkin("Hunyuan Video")

                mp4_files = [f for f in os.listdir('/root/HunyuanVideo/results') if f.endswith('.mp4')]
                for mp4_file in mp4_files:
                    try:
                        # Add it to Checkbin (!) 
                        path = os.path.join('/root/HunyuanVideo/results', mp4_file)
                        print(f"Found video at {path}!!")
                        checkbin.upload_file("inference_output", str(path), "video")
                        os.remove(path)
                        print(f"Deleted file: {mp4_file}")
                    except Exception as e:
                        print(f"Error deleting file {mp4_file}: {e}")

                checkbin.submit()


