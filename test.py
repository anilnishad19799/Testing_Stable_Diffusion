from utils import *
from text_embedding import get_text_embeds
from get_latent import produce_latents
from decode_latents import decode_img_latents

def prompt_to_img(prompts, height=512, width=512, num_inference_steps=50,
                  guidance_scale=7.5, latents=None):
  if isinstance(prompts, str):
    prompts = [prompts]

  # Prompts -> text embeds
  text_embeds = get_text_embeds(prompts)

  # Text embeds -> img latents
  latents = produce_latents(
      text_embeds, height=height, width=width, latents=latents,
      num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
  
  # Img latents -> imgs
  imgs = decode_img_latents(latents)

  return imgs


if __name__=='__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--input', '-input', type=str, default='Lion hunting deer')
  parser.add_argument('--img_size','-img_s', type=int, default=512)
  parser.add_argument('--itr','-itr', type=int, default=50)

  args = parser.parse_args()

  final_img = prompt_to_img(args['input'], args['img_size'], args['img_size'], args['itr'])[0]
  
  value = random.randint(0,1000000)
  path = os.getcwd().replace('\\','/')
  cv2.imwrite(f"{path}/{value}.png", value)

