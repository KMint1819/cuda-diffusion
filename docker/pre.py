from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

version = 'openai/clip-vit-large-patch14'
tokenizer = CLIPTokenizer.from_pretrained(version) 
transformer = CLIPTextModel.from_pretrained(version)

tokenizer.save_pretrained('/home/clip-vit-large-patch14')
transformer.save_pretrained('/home/clip-vit-large-patch14')