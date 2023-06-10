# Korosh Moosavi, 2023
# 
# This is intended to be used with a registry of emotes saved as a DataFrame of channels and their 
# emotes as pickle files in the same directory, as 'emotes_dict' for Twitch, and '[source]_dict' for others.
# Additionally, the file path to the trained W2V model is hard-coded and may need to be modified if using a different file structure.
#

import argparse
import gensim.models.word2vec
import numpy as np
import pickle
import warnings
from pprint import pprint
from .Tokenizer import TwitchTokenizer

warnings.filterwarnings('ignore')

def main():
    # Read in arguments for this chat instance
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", help="Channel where chat is taking place")
    parser.add_argument("--addons", nargs='*', help="Sources of emotes other than Twitch. Currently supported: bttv, ffz*, 7tv* [*Global emotes only]")
    parser.add_argument("--subs", nargs='*', help="List of channels the user is subscribed to")
    args = parser.parse_args()
    args.addons = [x.lower() for x in args.addons]

    # Modified tokenizer from EmoteControlled study source code on GitHub
    # Now accepts a list of emotes as an argument in order to not preprocess unrecognized emotes
    tokenizer = TwitchTokenizer()

    # Trained Word2Vec model containing embeddings for Twitch emotes
    model_dir = r"models\twitch_500_20e\model"
    model = gensim.models.word2vec.Word2Vec.load(model_dir)
    embeddings = model.wv
    del model   # We only need the learned embeddings

    def recommendEmotes(msg: str) -> list:
        """
        Returns a dict of the most similar emotes for each emote found in input.

        Param: msg - chat message as a string
        Param: subs - list of channels the user is subscribed to
        Param: channel - current channel the msg is being sent in
        Return: dict with detected emotes and the 3 most similar in 'channel'
        """
        rec = dict()
        tok = tokenizer.tokenize(text=msg,emotes=emotes_all)    

        # For each word in the chat message
        for token in tok:
            if token not in emotes_all:
                # Emote from unrecognized channel (or just text)
                continue

            # Calculate similarity between the current emote and each emote in the current channel
            sim_scores = {}
            for emote in emotes_ch:
                try:
                    sim_scores[emote] = embeddings.similarity(token.lower(), emote.lower())
                except Exception as e:
                    # No embedding data found for an emote
                    # Either embeddings are out of date or emote just isn't used enough
                    continue
                
            # Sort by similarity and return top 3 emotes
            sim_scores = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
            rec[token] = [emote for emote in sim_scores[:3]]

        # Return a dict of detected emotes as keys with recommendations as values
        return rec
    
    def find_channel(emote: str):
        """
        Search for the source channel of an emote and print it
        Emotes from non-Twitch sources may belong to multiple channels

        Param: emote - string of the emote being searched
        Return: Boolean - whether the input was found
        """
        # Label whether emote exists for each channel
        col_mask = emotes.isin([emote]).any()
        # Filter out channels labeled as True
        cols = col_mask[col_mask == True].index.tolist()

        # If not in a Twitch channel, search 3rd party extensions
        # Print all channels which use it, if found
        if len(cols) == 0:
            found = False
            if emote in bttv_emotes.values:
                found = True
                col_mask = bttv_emotes.isin([emote]).any()
                cols = col_mask[col_mask == True].index.tolist()
                print('BTTV:',cols)
            if emote in ffz_emotes.values:
                found = True
                col_mask = ffz_emotes.isin([emote]).any()
                cols = col_mask[col_mask == True].index.tolist()
                print('FFZ:',cols)
            if found:
                print()
            return found
        # Otherwise, print the Twitch channel the emote belongs to
        else:
            print(cols[0])
            return True
    
    # Assign global emotes
    base_twitch = ['GLOBAL_TWITCH']
    # 3rd party extensions
    if '7tv' in args.addons:
        print('Enabling 7TV emotes')
        base_twitch.extend(['GLOBAL_7TV'])

    # Load Twitch emotes
    with open('emote_dict', 'rb') as f:
        emotes = pickle.load(f)
        emotes.replace(['NaN', 'nan'], np.nan, inplace = True)

    # BTTV is large enough that it's handled separately
    if 'bttv' in args.addons:
        print('Enabling BTTV emotes')
        with open('bttv_dict', 'rb') as f:
            bttv_emotes = pickle.load(f)
            bttv_emotes.replace(['NaN', 'nan'], np.nan, inplace = True)

    # FFZ is smaller but still easier to manage alone
    if 'ffz' in args.addons:
        print('Enabling FFZ emotes')
        with open('ffz_dict', 'rb') as f:
            ffz_emotes = pickle.load(f).T
            ffz_emotes.replace(['NaN', 'nan'], np.nan, inplace = True)

    # Channel the user is currently chatting in
    channel = args.channel # Ex: 'HasanAbi'

    # Channels the user is actively subscribed to
    # Ex: subs = [
    #     'CohhCarnage',
    #     'KaiCenat',
    #     'forsen'
    # ]
    subs = args.subs
    twitch_subs = subs.copy()
    twitch_subs.extend(base_twitch)

    # Emotes native to the current channel
    emotes_ch = [x for x in emotes[channel].fillna('').values.flatten() if x != '']

    # Emotes the user has access to
    emotes_user = [x for x in emotes[twitch_subs].fillna('').values.flatten() if x != '']

    # Handle BTTV
    if 'bttv' in args.addons:
        bttv_subs = subs.copy()
        bttv_subs.extend(['GLOBAL_BTTV'])        
        emotes_ch.extend([x for x in bttv_emotes[channel].fillna('').values.flatten() if x != ''])
        emotes_user.extend([x for x in bttv_emotes[bttv_subs].fillna('').values.flatten() if x != ''])

    # Handle FFZ
    if 'ffz' in args.addons:
        ffz_subs = subs.copy()
        base_twitch.extend(['GLOBAL_FFZ'])
        if channel in ffz_emotes.columns: # Less channel support on FFZ than BTTV
            emotes_ch.extend([x for x in ffz_emotes[channel].fillna('').values.flatten() if x != ''])
            emotes_user.extend([x for x in ffz_emotes[ffz_subs].fillna('').values.flatten() if x != ''])

    # All relevant emotes for lookup
    emotes_all = emotes_ch + emotes_user

    msg = input('\nEnter your chat message:\n') 
    # Ex: "AYAYA here's a haHAA chat message forsenPls good luck cohhLUL remember to sub <3"

    # ?[command] format similar to Twitch chat's ![command]
    while True:
        if msg == '?exit':
            break

        # Search for string prepended by '?'
        # Full string is treated as an emote
        # Print both the source of the emote and the 5 most similar *tokens* (usually emotes)
        if msg.startswith('?'):
            found = find_channel(msg[1:])
            if found:
                print(embeddings.similar_by_key(msg[1:].lower(),topn=5))
            break

        rec_emotes = recommendEmotes(msg=msg)

        # See if any recommendations were found
        if sum(len(v) for v in rec_emotes.values()) > 0:
            print('\nHere are your recommended emotes:')
            pprint(rec_emotes)

            # Reprint the input with detected emotes replaced by recommendations
            print('\nHere is an example of how your message could look:')
            new_msg = msg
            for emote in rec_emotes.items():
                if len(emote[1]) > 0:
                    new_msg = new_msg.replace(emote[0],emote[1][0][0])
            print(new_msg)
        
        else:
            print('No emotes detected')

        # Prompt user for their next message/query
        msg = input('\nEnter your chat message:\n')


if __name__ == '__main__':
    main()

# Korosh Moosavi, 2023