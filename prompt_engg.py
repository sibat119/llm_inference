from datasets import Dataset
from src.utils.strings import replace_slot
from src.utils.files import load_yaml
from bs4 import BeautifulSoup

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

class PromptEngg:
    def __init__(self, config, level, dataset, llm="gpt-3.5-turbo", path_key='taxonomy_filepath', prompt_method='taxonomy', module_file='heu'):
        self.config = config
        self.level = level
        self.llm = llm

        if prompt_method == 'taxonomy':
            file_path = self.config[path_key]
            file_name = f"lv{self.level}.yaml"
            self.prompt_config = load_yaml(f"{file_path}/{file_name}")
            self.system_msg = self.prompt_config['system_role']['current']
            self.dataset = dataset.map(self.update_columns)
        
        
    def update_columns(self, ds):
        ds["level"] = self.level
        ds["llm"] = self.llm
        constraint_ds = Dataset.from_csv(self.config.get("constraint_data_path"))
        constraint = constraint_ds.filter(lambda x: x['id'] == ds['id'])['prompt'][0]
        try:
            title = ds["title"] if 'title' in ds.columns else ""
            
            body = ds["body"] if 'body' in ds.columns else ""
                
            answers = ds["answers"] if 'answers' in ds.columns else ""
            
            category = ds["Topic_Name"] if 'Topic_Name' in ds.columns else ""
                
        except:
            title = ds["title"] if 'title' in dict(ds).keys() else ""
            
            body = ds["body"] if 'body' in dict(ds).keys() else ""
            
            answers = ds["answers"] if 'answers' in dict(ds).keys() else "" 
            
            category = ds["Topic_Name"] if 'Topic_Name' in dict(ds).keys() else ""   
        
        directive = self.prompt_config['directive']['current']
        answers = (answers[0] if len(answers) > 0 else "") if isinstance(answers, list) else answers
        ds["system_role"] = replace_slot(
            self.system_msg, 
            {
                "title": remove_html_tags(title),
                "body" : remove_html_tags(body),
                "answers": remove_html_tags(answers),
                "category": category,
                "constraint": constraint,
            }
        )
        
        ds["directive"] = replace_slot(
            directive, 
            {
                "title": remove_html_tags(title),
                "body" : remove_html_tags(body),
                "answers": remove_html_tags(answers),
                "category": category,
                "constraint": constraint,
            }
        )
        
        gen_message = self.prompt_config['gen_message']
        ds["gen_message"] = replace_slot(
            gen_message, 
            {
                "title": remove_html_tags(title),
                "body" : remove_html_tags(body),
                "answers": remove_html_tags(answers),
                "category": category,
                "constraint": constraint,
            }
        )
        
        # breakpoint()
        return ds
    
    
    def get_updated_dataset(self):
        return self.dataset
