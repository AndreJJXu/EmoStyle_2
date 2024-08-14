# python3 optimization/run_optimization_comprehensive.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/data/test.wav" --ckpt ./pretrained_models/landscape.pt  --stylegan_size 512 --emotion "positive"
CUDA_VISIBLE_DEVICES=1 python3 optimization/run_optimization_comprehensive.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt  --stylegan_size 512 --emotion "positive" --results_dir /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/Experience/Semantic/raining  --description "raining day"
# python3 optimization/StyleCLIP.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt  --stylegan_size 512 --emotion "positive" --results_dir /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/Experience/Semantic/raining  --description "raining" 
# python3 optimization/Lee.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt  --stylegan_size 512 --emotion "positive" --results_dir /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/Experience/Semantic/raining  --description "raining"
# python3 optimization/run_optimization_wo_music.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt  --stylegan_size 512 --emotion "positive" --results_dir /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/Experience/Semantic/raining  --description "raining"
# python3 optimization/run_optimization_wo_audio.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt  --stylegan_size 512 --emotion "positive" --results_dir /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/Experience/Semantic/raining  --description "raining"
# python3 optimization/run_optimization_wo_text.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "/mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt  --stylegan_size 512 --emotion "positive" --results_dir /mnt/ssd/BeautifulXJJ/AIGC/Sound-Image-Generation/sound-guided-semantic-image-manipulation/Experience/Semantic/raining  --description "raining"

























# python3 optimization/run_optimization_expanded.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/explosion.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 512 
# python3 optimization/run_optimization_expanded_2.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/fire.wav" --ckpt ./pretrained_models/landscape.pt
# python3 optimization/run_optimization_expanded_2.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/explosion.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 512 --description "fire mountain"
# python3 -m pdb optimization/run_optimization_expanded.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/explosion.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 512 --description "beautiful smile"
#landscape
# python3 optimization/run_optimization_comprehensive.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 512 --description "happy smile"
# python3 optimization/run_optimization_expanded.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/explosion.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 512 
# python3 optimization/run_optimization_expanded_2.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/raining.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 512 
# python3 optimization/run_optimization_expanded_2.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/explosion.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 512 --description "fire mountain"