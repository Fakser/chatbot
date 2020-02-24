import time
import random
import os
import numpy as np
from seq2seqlib import Seq2seq
import praw #this for reddit memes
from flask import Flask, request
from pymessenger.bot import Bot


seq2seq = Seq2seq() #initializing seq2seq model for chatting

if len(list(os.listdir(seq2seq.checkpoint_dir))) < 3:
    seq2seq.train(30)
        
else:
    print('model loaded from last checkpoint')

      
seq2seq.load()

#for reddit memes we need to initialize connection with reddit

reddit = praw.Reddit(client_id = '', # paste here your client id
                     client_secret = '', # paste here your client secret
                     user_agent = "", # paste here your user agent
                     username = '', # paste here your REDDIT username
                     password = '') # paste here your REDDIT password 

subreddits = ['dataisbeautifull','memes', 'blursedimages', 'ItemShop','MemeEconomy','wallstreetbets']

app = Flask(__name__)
FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'
VERIFY_TOKEN = ''# <paste your verify token here>
PAGE_ACCESS_TOKEN = ''# paste your page access token here>"

bot = Bot(PAGE_ACCESS_TOKEN)

#We will receive messages that Facebook sends our bot at this endpoint 
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook.""" 
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
       output = request.get_json()
       for event in output['entry']:
          messaging = event['messaging']
          for message in messaging:
            if message.get('message'):
                #Facebook Messenger ID for user so we know where to send response back to
                recipient_id = message['sender']['id']
                if message['message'].get('text'):
                    
                    response_sent_text = get_message(message['message'].get('text'))
                    send_message(recipient_id, response_sent_text)
                #if user sends us a GIF, photo,video, or any other non-text item
                if message['message'].get('attachments'):
                    response_sent_nontext = get_message()
                    send_message(recipient_id, response_sent_nontext)
    return "Message Processed"


def verify_fb_token(token_sent):
    #take token sent by facebook and verify it matches the verify token you sent
    #if they match, allow the request, else return an error 
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


#chooses a random message to send to the user
def get_message(string = None):
    if string != None:
        if np.random.uniform(0,1) < 0.9: 
            return seq2seq.chat(string)
        else:
            return random.choice([submission.url for submission in reddit.subreddit(random.choice(subreddits)).hot(limit=100)])
    else:
        return random.choice([submission.url for submission in reddit.subreddit(random.choice(subreddits)).hot(limit=100)])
    

#uses PyMessenger to send response to user
def send_message(recipient_id, response):
    #sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    print(response)
    return "success"

if __name__ == "__main__":
    app.run()