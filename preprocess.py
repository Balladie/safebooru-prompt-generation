import os
import sys
import pandas as pd
from tqdm import tqdm


def filter_by_tags_0(text, tags_0):
    tags = text.split(', ')
    tags = [tag.replace(' ', '_') for tag in tags]

    tags = [tag.replace('(', '').replace(')', '') for tag in tags]
    tags_0 = [tag.replace('(', '').replace(')', '') for tag in tags_0]

    tags = [tag for tag in tags if tag in tags_0]

    tags = [tag.replace('_', ' ') for tag in tags]

    return ', '.join(tags)


if __name__ == '__main__':
    prompts_path = sys.argv[1]
    prompts_fn = os.path.basename(prompts_path)
    prompts_dir = os.path.dirname(prompts_path)

    out_fn = prompts_fn.split('.')[0] + '_filtered_v1.' + prompts_fn.split('.')[1]
    out_path = os.path.join(prompts_dir, out_fn)

    tags_path = 'data/danbooru.csv'
    quality_tags_path = 'data/extra-quality-tags.csv'

    # read files
    prompts_data = pd.read_csv(prompts_path)
    tags_data = pd.read_csv(tags_path, header=None)
    quality_tags_data = pd.read_csv(quality_tags_path)

    # filter out unsafe prompts and NaNs
    safe_prompts_data = prompts_data['safebooru_clean'].dropna()

    # Danbooru tags with category 0
    tags_data = tags_data[tags_data[1] == 0]
    tags_0 = list(tags_data[0])

    # filter out categories except for category 0
    for i in tqdm(safe_prompts_data.index):
        safe_prompts_data.at[i] = filter_by_tags_0(safe_prompts_data.at[i], tags_0)

    # save preprocessed .csv file
    safe_prompts_data.to_csv(out_path, index=False)