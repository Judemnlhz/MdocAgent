import json
import re
from dataclasses import dataclass
from PIL import Image
import os
import pymupdf
from tqdm import tqdm
from datetime import datetime
import glob

@dataclass
class Content:
    image: Image
    image_path: str
    txt: str
    
class BaseDataset():
    def __init__(self, config):
        self.config = config
        self.IM_FILE = lambda doc_name,index: f"{self.config.extract_path}/{doc_name}_{index}.png"
        self.TEXT_FILE = lambda doc_name,index: f"{self.config.extract_path}/{doc_name}_{index}.txt"
        self.EXTRACT_DOCUMENT_ID = lambda sample: re.sub("\\.pdf$", "", sample["doc_id"]).split("/")[-1] 
        current_time = datetime.now()
        self.time = current_time.strftime("%Y-%m-%d-%H-%M")
        self._baec_debug_printed = 0
    
    def load_data(self, use_retreival=True):
        path = self.config.sample_path
        if use_retreival:
            try:
                assert(os.path.exists(self.config.sample_with_retrieval_path))
                path = self.config.sample_with_retrieval_path
            except:
                print("Use original sample path!")
                
        assert(os.path.exists(path))
        with open(path, 'r') as f:
            samples = json.load(f)
            
        return samples
    
    def dump_data(self, samples, use_retreival=True):
        if use_retreival:
            path = self.config.sample_with_retrieval_path
        else:
            path = self.config.sample_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        
        return path
    
    def load_latest_results(self):
        print(self.config.result_dir)
        path = find_latest_json(self.config.result_dir)
        with open(path, 'r') as f:
            samples = json.load(f)
        return samples, path
    
    def dump_reults(self, samples):
        os.makedirs(self.config.result_dir, exist_ok=True)
        path = os.path.join(self.config.result_dir, self.time + ".json")
        with open(path, 'w') as f:
            json.dump(samples, f, indent = 4)
        return path
    
    def load_retrieval_data(self):
        assert(os.path.exists(self.config.sample_with_retrieval_path))
        with open(self.config.sample_with_retrieval_path, 'r') as f:
            samples = json.load(f)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_retrieval_data(sample)
        return samples
    
    def load_sample_retrieval_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question:str = sample[self.config.question_key]
        texts = []
        images = []
        if self._use_baec_sample(sample):
            selected_pages, used_k = self._get_baec_selected_pages(sample)
            self._print_baec_debug(selected_pages, used_k)
            for page in selected_pages:
                try:
                    page = int(page)
                except (TypeError, ValueError):
                    continue
                if page < 0 or page >= len(content_list):
                    continue
                texts.append(content_list[page].txt.replace("\n", ""))
                images.append(content_list[page].image_path)
            return question, texts, images

        if self.config.use_mix:
            if self.config.r_mix_key in sample:
                for page in sample[self.config.r_mix_key][:self.config.top_k]:
                    if page in sample[self.config.r_image_key]:
                        origin_image_path = ""
                        origin_image_path = content_list[page].image_path
                        images.append(origin_image_path)
                    if page in sample[self.config.r_text_key]:
                        texts.append(content_list[page].txt.replace("\n", ""))
        else:
            if self.config.r_text_key in sample:
                for page in sample[self.config.r_text_key][:self.config.top_k]:
                    texts.append(content_list[page].txt.replace("\n", ""))
            if self.config.r_image_key in sample:
                for page in sample[self.config.r_image_key][:self.config.top_k]:
                    origin_image_path = ""
                    origin_image_path = content_list[page].image_path
                    images.append(origin_image_path)
                        
        return question, texts, images

    def _use_baec_sample(self, sample):
        if not bool(getattr(self.config, "use_baec", False)):
            return False
        return "baec_stage1" in sample or "baec_trace" in sample

    def _get_baec_selected_pages(self, sample):
        stage1 = sample.get("baec_stage1") or {}
        trace = sample.get("baec_trace") or {}
        selected_pages = stage1.get("selected_pages")
        if selected_pages is None:
            selected_pages = trace.get("selected_pages", [])
        used_k = stage1.get("used_k", trace.get("used_k", len(selected_pages)))
        k_max = getattr(self.config, "baec_k_max", self.config.top_k)
        return selected_pages[:k_max], used_k

    def _print_baec_debug(self, selected_pages, used_k):
        debug_limit = int(getattr(self.config, "baec_debug_samples", 0) or 0)
        if self._baec_debug_printed >= debug_limit:
            return
        print(f"[BAEC] selected_pages={selected_pages}, used_k={used_k}")
        self._baec_debug_printed += 1
    
    def load_full_data(self):
        samples = self.load_data(use_retreival=False)
        for sample in tqdm(samples):
            _, sample["texts"], sample["images"] = self.load_sample_full_data(sample)
        return samples
    
    def load_sample_full_data(self, sample):
        content_list = self.load_processed_content(sample, disable_load_image=True)
        question:str = sample[self.config.question_key]
        texts = []
        images = []
        
        if self.config.page_id_key in sample:
            sample_no_list = sample[self.config.page_id_key]
        else:
            sample_no_list = [i for i in range(0,min(len(content_list),self.config.vlm_max_page))]
        for page in sample_no_list:
            texts.append(content_list[page].txt.replace("\n", ""))
            origin_image_path = ""
            origin_image_path = content_list[page].image_path
            images.append(origin_image_path)
                        
        return question, texts, images
      
    def load_processed_content(self, sample: dict, disable_load_image=True)->list[Content]:
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        content_list = []
        for page_idx in range(self.config.max_page):
            im_file = self.IM_FILE(doc_name, page_idx)
            text_file = self.TEXT_FILE(doc_name, page_idx)
            if not os.path.exists(im_file):
                break
            img = None
            if not disable_load_image:
                img = self.load_image(im_file)
            txt = self.load_txt(text_file)
            content_list.append(Content(image=img, image_path=im_file, txt=txt)) 
        return content_list
    
    def load_image(self, file):
        pil_im = Image.open(file)
        return pil_im

    def load_txt(self, file):
        max_length = self.config.max_character_per_page
        with open(file, 'r') as file:
            content = file.read()
        content = content.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return content[:max_length]
    
    def extract_content(self, resolution=144):
        samples = self.load_data()
        for sample in tqdm(samples):
            self._extract_content(sample, resolution=resolution)
            
    def _extract_content(self, sample, resolution=144):
        max_pages=self.config.max_page
        os.makedirs(self.config.extract_path, exist_ok=True)
        image_list = list()
        text_list = list()
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        with pymupdf.open(os.path.join(self.config.document_path, sample["doc_id"])) as pdf:
            for index, page in enumerate(pdf[:max_pages]):
                # save page as an image
                im_file = self.IM_FILE(doc_name,index)
                if not os.path.exists(im_file):
                    im = page.get_pixmap(dpi=resolution)
                    im.save(im_file)
                image_list.append(im_file)
                # save page text
                txt_file = self.TEXT_FILE(doc_name,index)
                if not os.path.exists(txt_file):
                    text = page.get_text("text")
                    with open(txt_file, 'w') as f:
                        f.write(text)
                text_list.append(txt_file)
                
        return image_list, text_list
    
def extract_time(file_path):
    file_name = os.path.basename(file_path)
    time_str = file_name.split(".json")[0]
    return datetime.strptime(time_str, "%Y-%m-%d-%H-%M")

def find_latest_json(result_dir):
    pattern = os.path.join(result_dir, "*-*-*-*-*.json")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith('_results.json')]
    if not files:
        print(f"Json file not found at {result_dir}")
        return None
    latest_file = max(files, key=extract_time)
    return latest_file
