import glob, os
import IPython

output_path = "train_dir"
ckpts = sorted([f for f in glob.glob(output_path+"/*/*.pth")])
configs = sorted([f for f in glob.glob(output_path+"/*/*.json")])

if ckpts and configs:
    model_path = "train_dir/vits_vctk-February-10-2025_06+26PM-0e7bb00/checkpoint_1220000.pth"
    config_path = "train_dir/vits_vctk-February-10-2025_06+26PM-0e7bb00/config.json"

    cmd = f'''tts --text "Един тежък крак зашари за ключа, достигна го и светлината заля всичко. Eднообразната светлина на прозаичното изкуствено слънце." \
        --model_path "{model_path}" \
        --config_path "{config_path}" \
        --speaker_id 1 \
        --out_path "out.wav"'''
    
    os.system(cmd)
else:
    print("No checkpoint or config files found.")

IPython.display.Audio("out.wav", autoplay=True)
