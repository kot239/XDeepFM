import pandas as pd
import numpy as np


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan


def isrc_to_country(isrc):
    if type(isrc) == str:
         return isrc[0:2]
    else:
        return np.nan


def return_number_played(x, dict_count_song_played_train, dict_count_song_played_test):
    try:
        return dict_count_song_played_train[x]
    except KeyError:
        try:
            return dict_count_song_played_test[x]
        except KeyError:
            return 0


def return_user_activity(x, dict_user_activity):
    try:
        return dict_user_activity[x]
    except KeyError:
        return 0


def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0


def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')


def genres_count(x):
    if type(x) != str:
        return 1
    else:
        return 1 + x.count('|')


def get_train_and_test_df():
    members = pd.read_csv('members.csv')
    songs = pd.read_csv('songs.csv')
    test = pd.read_csv('test.csv')
    train = pd.read_csv('train.csv')
    song_extra_info = pd.read_csv('song_extra_info.csv')

    songs = songs.merge(song_extra_info, how='left', on='song_id')

    members_num = len(members)
    songs_num = len(songs)

    # subset of members (users) and songs
    members_subs = members.sample(n=members_num // 2)
    songs_subs = songs.sample(n=songs_num // 10)

    # merge songs and users
    train_subs = train.merge(members_subs, how='inner', on='msno')
    train_subs.rename(columns = {'msno' : 'user_id'}, inplace = True)
    train_subs = train_subs.merge(songs_subs, how='inner', on='song_id')

    test_subs = test.merge(members_subs, how='inner', on='msno')
    test_subs.rename(columns = {'msno' : 'user_id'}, inplace = True)
    test_subs = test_subs.merge(songs_subs, how='inner', on='song_id')

    # add median age to feature
    median_age = train_subs[train_subs['bd'] != 0]['bd'].median()
    train_subs['bd'] = np.where((train_subs.bd == 0), median_age, train_subs.bd)
    test_subs['bd'] = np.where((test_subs.bd == 0), median_age, test_subs.bd)

    # add song year to dataframe
    train_subs['song_year'] = train_subs['isrc'].apply(isrc_to_year)
    median_song_year = train_subs[train_subs['song_year'] != None]['song_year'].median()
    train_subs['song_year'] = train_subs['song_year'].fillna(median_song_year)

    test_subs['song_year'] = test_subs['isrc'].apply(isrc_to_year)
    test_subs['song_year'] = test_subs['song_year'].fillna(median_song_year)

    # add song country to dataframe
    train_subs['song_country'] = train_subs['isrc'].apply(isrc_to_country)
    test_subs['song_country'] = test_subs['isrc'].apply(isrc_to_country)

    # add expiration date to dataframe
    median_expiration_date = train_subs[train_subs['expiration_date'] != 0]['expiration_date'].median()
    train_subs['expiration_date'] = np.where((train_subs.expiration_date == 0), median_expiration_date, train_subs.expiration_date)

    median_expiration_date_test = test_subs[test_subs['expiration_date'] != 0]['expiration_date'].median()
    test_subs['expiration_date'] = np.where((test_subs.expiration_date == 0), median_expiration_date_test, test_subs.expiration_date)

    # add regestration date to dataframe
    median_registration_init_time = train_subs[train_subs['registration_init_time'] != 0]['registration_init_time'].median()
    train_subs['registration_init_time'] = np.where((train_subs.registration_init_time == 0), median_registration_init_time, train_subs.registration_init_time)

    median_registration_init_time_test = test_subs[test_subs['registration_init_time'] != 0]['registration_init_time'].median()
    test_subs['registration_init_time'] = np.where((test_subs.registration_init_time == 0), median_registration_init_time_test, test_subs.registration_init_time)

    # change expiration date and regestration date to membership days
    train_subs['membership_days'] = train_subs['expiration_date'].subtract(train_subs['registration_init_time']).astype(int)
    train_subs.drop(columns = ['registration_init_time' , 'expiration_date'] , inplace = True)

    test_subs['membership_days'] = test_subs['expiration_date'].subtract(test_subs['registration_init_time']).astype(int)
    test_subs.drop(columns = ['registration_init_time' , 'expiration_date'] , inplace = True)

    # drop excess column
    train_subs.drop(['isrc'], axis = 1, inplace = True)
    test_subs.drop(['isrc'], axis = 1, inplace = True)

    # add number of times when song was played
    dict_count_song_played_train = {k: v for k, v in train_subs['song_id'].value_counts().iteritems()}
    dict_count_song_played_test = {k: v for k, v in test_subs['song_id'].value_counts().iteritems()}

    train_subs['number_of_time_played'] = train_subs['song_id'].apply(lambda x: return_number_played(x, dict_count_song_played_train, dict_count_song_played_test))
    test_subs['number_of_time_played'] = test_subs['song_id'].apply(lambda x: return_number_played(x, dict_count_song_played_train, dict_count_song_played_test))

    # add user activity to dataframe
    dict_user_activity = {k: v for k,v in pd.concat([train_subs['user_id'], test_subs['user_id']], axis = 0).value_counts().iteritems()}

    train_subs['user_activity'] = train_subs['user_id'].apply(lambda x: return_user_activity(x, dict_user_activity))
    test_subs['user_activity'] = test_subs['user_id'].apply(lambda x: return_user_activity(x, dict_user_activity))

    # add lyricist to dataframe
    train_subs['lyricist'] = train_subs['lyricist'].astype('category')
    train_subs['lyricist'] = train_subs['lyricist'].cat.add_categories(['no_lyricist'])
    train_subs['lyricist'].fillna('no_lyricist', inplace=True)
    train_subs['lyricists_count'] = train_subs['lyricist'].apply(lyricist_count).astype(np.int8)

    test_subs['lyricist'] = test_subs['lyricist'].astype('category')
    test_subs['lyricist'] = test_subs['lyricist'].cat.add_categories(['no_lyricist'])
    test_subs['lyricist'].fillna('no_lyricist', inplace=True)
    test_subs['lyricists_count'] = test_subs['lyricist'].apply(lyricist_count).astype(np.int8)

    # add composer to dataframe
    train_subs['composer'] = train_subs['composer'].astype('category')
    train_subs['composer'] = train_subs['composer'].cat.add_categories(['no_composer'])
    train_subs['composer'].fillna('no_composer', inplace=True)
    train_subs['composer_count'] = train_subs['composer'].apply(composer_count).astype(np.int8)

    test_subs['composer'] = test_subs['composer'].astype('category')
    test_subs['composer'] = test_subs['composer'].cat.add_categories(['no_composer'])
    test_subs['composer'].fillna('no_composer', inplace=True)
    test_subs['composer_count'] = test_subs['composer'].apply(composer_count).astype(np.int8)

    # add if artist, composer and lyricist is the same person
    train_subs['artist_composer_lyricist'] = ((np.asarray(train_subs['artist_name']) == np.asarray(train_subs['composer'])) & 
                                        np.asarray((train_subs['artist_name']) == np.asarray(train_subs['lyricist'])) & 
                                        np.asarray((train_subs['composer']) == np.asarray(train_subs['lyricist']))).astype(np.int8)

    test_subs['artist_composer_lyricist'] = ((np.asarray(test_subs['artist_name']) == np.asarray(test_subs['composer'])) & 
                                        np.asarray((test_subs['artist_name']) == np.asarray(test_subs['lyricist'])) & 
                                        np.asarray((test_subs['composer']) == np.asarray(test_subs['lyricist']))).astype(np.int8)

    # add artist name to dataframe and if the artist is featured with someone
    train_subs['artist_name'] = train_subs['artist_name'].astype('category')
    train_subs['artist_name'] = train_subs['artist_name'].cat.add_categories(['no_artist'])
    train_subs['artist_name'].fillna('no_artist', inplace=True)
    train_subs['is_featured'] = train_subs['artist_name'].apply(is_featured).astype(np.int8)

    test_subs['artist_name'] = test_subs['artist_name'].astype('category')
    test_subs['artist_name'] = test_subs['artist_name'].cat.add_categories(['no_artist'])
    test_subs['artist_name'].fillna('no_artist', inplace=True)
    test_subs['is_featured'] = test_subs['artist_name'].apply(is_featured).astype(np.int8)

    # add the number of artist songs
    train_subs['artist_count'] = train_subs['artist_name'].apply(artist_count).astype(np.int8)
    test_subs['artist_count'] = test_subs['artist_name'].apply(artist_count).astype(np.int8)

    # add if artist and composer is the same person
    train_subs['artist_composer'] = (np.asarray(train_subs['artist_name']) == np.asarray(train_subs['composer'])).astype(np.int8)
    test_subs['artist_composer'] = (np.asarray(test_subs['artist_name']) == np.asarray(test_subs['composer'])).astype(np.int8)

    # add number of generes to dataframe
    train_subs['number_of_genres'] = train_subs['genre_ids'].apply(genres_count)
    test_subs['number_of_genres'] = test_subs['genre_ids'].apply(genres_count)

    return train_subs, test_subs
