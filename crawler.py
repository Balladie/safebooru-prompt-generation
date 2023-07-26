import os
import pandas as pd
from pybooru import Danbooru


def print_skip_image(image, miss):
    if 'id' in image.keys():
        print(f'Skipping image id without {miss}:', image['id'])
    else:
        print(f'Skipping image id without {miss} (no id)')


if __name__ == '__main__':
    data_dir = 'data'
    output_fn = 'safebooru_prompts_2023.csv'
    csv_path = os.path.join(data_dir, output_fn)

    column_name = 'safebooru_clean'

    limit = 1000
    page = 1
    page_limit = 0
    resume_page = 1001

    # Pandas dataframe
    if resume_page and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        page = resume_page
    else:
        df = pd.DataFrame(columns=[column_name])

    # Create a client object for the booru site
    booru = Danbooru(site_name='safebooru')

    while True:
        print(f"Getting {page}'th page, total {len(df)} tags collected", end='\r')

        # Get the images from the current page
        try:
            images = booru.post_list(limit=limit, page=page)
        except Exception as e:
            print()
            print(str(e))
            print(f'--> Skipping page at {page}')

            # Break the loop if page_limit is achieved
            if page_limit > 0 and page >= page_limit:
                break
            else:
                page += 1
            
            continue


        # Break the loop if no more images are returned
        if not images:
            break

        # Append the tags to the list
        for image in images:
            if 'score' not in image.keys():
                print_skip_image(image, miss='score')
                continue

            if 'tag_string_general' not in image.keys():
                print_skip_image(image, miss='tag_string_general')
                continue

            if image['score'] >= 8:
                prompt = image['tag_string_general'].replace(' ', ', ').replace('_', ' ')
                row_to_add = pd.DataFrame.from_dict({column_name: prompt}, orient='index', columns=[column_name])
                df = pd.concat([df, row_to_add], ignore_index=True)

        # Save backup csv file
        if page % 100 == 0:
            df.to_csv(csv_path, index=False)

        # Break the loop if page_limit is achieved
        if page_limit > 0 and page >= page_limit:
            break
        else:
            page += 1

    print()
    print('Total number of images collected:', len(df))

    df.to_csv(csv_path, index=False)