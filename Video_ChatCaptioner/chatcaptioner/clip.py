import open_clip

class ClipScore():
    def __init__(self, device='cuda:0'):
        # load open clip to device
        self.device = device
        clip, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.clip = clip.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        
        
    def clip_IT_score(self, image, texts):
        '''
        compute the average similarity score of a given image and a list of texts
        '''
        if isinstance(texts, str):
            texts = [texts]
        image = self.clip_preprocess(image)[None].to(self.device)
        texts = self.clip_tokenizer(texts).to(self.device)
        with torch.no_grad():
            image_f = self.clip.encode_image(image).float()
            texts_f = self.clip.encode_text(texts).float()
            image_f /= image_f.norm(dim=-1, keepdim=True)
            texts_f /= texts_f.norm(dim=-1, keepdim=True)
            similarity = (image_f.cpu().numpy() @ texts_f.cpu().numpy().T).mean()
            similarity = round(float(similarity), 3)
        return similarity