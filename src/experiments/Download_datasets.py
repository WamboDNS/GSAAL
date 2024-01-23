import ssl
import os
import fsspec
from tqdm import tqdm
import json
import wget

class Utils_local():
    """Changes the save path of the datasets to a local path

    This new class changes the download datasets function so it downloads all dataset in the repository, instead that in the 
    package environment file. Class function was taken from the class Utils from the AD Bench package. Reason as to why not to
    use AD Bench directly is because of a dependency issue with the package and our build of CUDA.
    """

    
    def download_datasets(self, repo='jihulab'):
        ssl._create_default_https_context = ssl._create_unverified_context
        print('if there is any question while downloading datasets, we suggest you to download it from the website:')
        print('https://github.com/Minqi824/ADBench/tree/main/adbench/datasets')
        print('如果您在中国大陆地区，请使用链接：')
        print('https://jihulab.com/BraudoCC/ADBench_datasets/')
        # folder_list = ['CV_by_ResNet18', 'CV_by_ViT', 'NLP_by_BERT', 'NLP_by_RoBERTa', 'Classical']
        folder_list = ['CV_by_ResNet18', 'NLP_by_BERT', 'Classical']
        
        if repo == 'github':
            fs = fsspec.filesystem("github", org="Minqi824", repo="ADBench")
            print(f'Downloading datasets from the remote github repo...')
            for folder in tqdm(folder_list):
                save_path = os.path.join('datasets', folder)
                print(f'Current saving path: {save_path}')
                if os.path.exists(save_path):
                    print(f'{folder} already exists. Skipping download...')
                    continue

                os.makedirs(save_path, exist_ok=True)
                fs.get(fs.ls("datasets/" + folder), save_path, recursive=True)
        
        elif repo == 'jihulab':
            print(f'Downloading datasets from jihulab...')
            url_repo = 'https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/339d2ab2d53416854f6535442a67393634d1a778'
            # load the datasets path
            url_dictionary = url_repo + '/datasets_files_name.json'
            wget.download(url_dictionary,out = './datasets_files_name.json')
            with open('./datasets_files_name.json', 'r') as json_file:
                loaded_dict = json.loads(json_file.read())

            # download datasets
            for folder in tqdm(folder_list):
                datasets_list = loaded_dict[folder]
                save_fold_path = os.path.join('datasets', folder)
                if os.path.exists(save_fold_path) is False:
                    os.makedirs(save_fold_path, exist_ok=True)
                for datasets in datasets_list:
                    save_path = os.path.join(save_fold_path, datasets)
                    if os.path.exists(save_path):
                        print(f'{datasets} already exists. Skipping download...')
                        continue
                    print(f'Current saving path: {save_path}')
                    # url = os.path.join(url_repo,folder,datasets)
                    url = f'{url_repo}/{folder}/{datasets}'
                    wget.download(url,out = save_path)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    utils = Utils_local()
    utils.download_datasets(repo='jihulab')