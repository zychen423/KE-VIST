import os
import json
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

row_template =  """
                        <tr>
                            {}
                        </tr>
"""
image_template= """
                            <td>
                                <img src="{}">
                            </td>
"""
text_template= """
                            <td>
                                {}
                            </td>
"""
class StoryReader():
    def __init__(self, max_story_len=10, file_name='index.html'):
        self.max_story_len = max_story_len
        self.file_name = file_name
        self.html_dir = '/home/joe32140/public_html/visual_storytelling/'
        self.template_path = '/home/bendan0617/Visual-Storytelling/template/template.html'
        self.id2idx = json.load(open('/home/joe32140/data/VIST/dii/id2idx.json'))
        self.dii = json.load(open('/home/joe32140/data/VIST/dii/val.description-in-isolation.json'))
        self.image_blocks = []
        self.gen_stories = []
        self.gt_stories = []
        self.imgs_att=[]
        self.gt_template = []
        self.predict_event = []
        self.event_gt = []
        self.url_root = 'data/val-resized/'
        self.att_url_root = 'data/att_weight/'
        self.count=0
        self.container = {}
        self.sorted_keys =[]
        self.old_temp=""

    def load_content2container(self, content, type, row_name):
        # print("content", content)
        single_content = {"content":content, "type":type}
        if row_name in self.container:
            self.container[row_name].append(single_content)
        else:
            self.container[row_name] = []
            self.container[row_name].append(single_content)
            self.sorted_keys.append(row_name)

    def dump_html(self):
        with open(self.template_path, 'r', encoding='utf-8') as f:
            html = f.read()
        temp = ""
        content_number = len(self.container)

        # assert
        length = 0
        for key in self.container.keys():
            if length ==0:
                length = len(self.container[key])
            else:
                assert len(self.container[key])== length
        # print("keys",self.container.keys())
        #print("Assert Ok")
        for i in range(len(self.container['Image'])):
            for j in self.sorted_keys:
                if self.container[j][i]['type']=="text":
                    temp += self.build_text_html(self.container[j][i]["content"], j)
                elif self.container[j][i]['type']=="image":
                    temp += self.build_img_html(self.container[j][i]["content"], self.url_root)
                    temp += self.build_dii_html(self.container[j][i]["content"])



        temp = temp + self.old_temp
        html = html.format(temp)
        self.old_temp = temp
        self.container = {} # Delete the batch data which is  in the container
        with open(os.path.join(self.html_dir, self.file_name), 'w', encoding='utf-8') as outfile:
            outfile.write(html)
        #print("Finish writing")

    def build_img_html(self, images, root):
        temp=""
        temp += text_template.format("Image")
        for img_path in images:
            temp += image_template.format(root+img_path)
        return row_template.format(temp)

    def build_text_html(self, story, text):
        temp=""
        temp += text_template.format(text)
        for sen in story:
            temp += text_template.format(sen)
        return row_template.format(temp)

    def build_dii_html(self, images):
        temp=""
        temp += text_template.format("Description")
        for img in images:
            try:
                idx = self.id2idx[img[:-4]]
                temp += text_template.format(self.dii['annotations'][idx][0]['text'])
            except:
                temp += text_template.format("None")
        return row_template.format(temp)


def main():
    reader = StoryReader()
    root='/home/joe32140/data/VIST/images/val-resized/'
    text_path='/home/bendan0617/Visual-Storytelling/data/sis/modified_val.story-in-sequence.json'
    dialogs = json.load(open(text_path, 'r'))['annotations']
    for index in range(100):
        description = []
        image_paths = []
        for i in range(5):
            description.append(dialogs[index*5+i][0]['text'])
            image_id = dialogs[index*5+i][0]['photo_flickr_id']
            if os.path.exists(os.path.join(root, image_id+'.jpg')):
                image_paths.append(image_id+'.jpg')
            elif os.path.exists(os.path.join(root, image_id+'.png')):
                image_paths.append(image_id+'.png')
            else:
                print("NO")
                continue
        reader.load_img(image_paths)
        reader.load_gen_story(description)
        reader.load_gt_story(description)
    reader.dump_html()
if __name__ == '__main__':
    main()
