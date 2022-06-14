from torch.utils.data import Dataset


class KKBoxDataset(Dataset):
    def __init__(self, samples, results):
        self.data = samples
        self.results = results
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        item = {}
        # user data (6 features)
        # 'city', 'bd', 'gender', 'registered_via', 'membership_days', 'user_activity'
        item['city'] = row[0]
        item['bd'] = row[1]
        item['gender'] = row[2]
        item['registered_via'] = row[3]
        item['membership_days'] = row[12]
        item['user_activity'] = row[14]

        # song data (3 features)
        # 'language', 'source_system_tab', 'source_screen_name', 'source_type',
        # 'song_year', 'song_country', 'genre_ids', 'song_length',
        # 'number_of_time_played', 'lyricists_count', 'composer_count', 
        # 'artist_count', 'number_of_genres', 'artist_composer_lyricist',
        # 'is_featured', 'artist_composer'
        item['language'] = row[4]
        item['source_system_tab'] = row[5]
        item['source_screen_name'] = row[6]
        item['source_type'] = row[7]
        item['song_year'] = row[8]
        item['song_country'] = row[9]
        item['genre_ids'] = row[10]
        item['song_length'] = row[11]
        item['number_of_time_played'] = row[13]
        item['lyricists_count'] = row[15]
        item['composer_count'] = row[16]
        item['artist_count'] = row[17]
        item['number_of_genres'] = row[18]
        item['artist_composer_lyricist'] = row[19]
        item['is_featured'] = row[20]
        item['artist_composer'] = row[21]

        # target (1 feature)
        item['target'] = self.results[idx]
        return item