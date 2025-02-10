import glob, os
import IPython

output_path = "train_dir"
ckpts = sorted([f for f in glob.glob(output_path+"/*/*.pth")])
configs = sorted([f for f in glob.glob(output_path+"/*/*.json")])

if ckpts and configs:
    # model_path = ckpts[-1]
    # config_path = configs[-1]
    model_path = "train_dir/vits_vctk-February-09-2025_11+40AM-5956d67/checkpoint_750000.pth"
    config_path = "train_dir/vits_vctk-February-09-2025_11+40AM-5956d67/config.json"
    os.system(f"tts --text 'Православната църква почита днес паметта на свети мъченик Трифон. Денят, известен в народната традиция като Трифон Зарезан, дълги години е отбелязван по стар стил на 14 февруари и е свързан със зарязването на лозята преди идването на пролетта. Преместването му две седмици по-рано се дължи на преминаването към Новоюлианския календар през 1968 г. Понастоящем на 14 февруари православните християни почитат успението на св. Кирил' \
        --model_path {model_path} \
        --config_path {config_path} \
        --speaker_id 1 \
        --out_path out.wav")
else:
    print("No checkpoint or config files found.")

IPython.display.Audio("out.wav")
