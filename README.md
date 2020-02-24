# chatbot
Chatbot that is supposed to imitate me.


Because of RODO i cannot add my own messanger data as it contains others people personal information. 
You can train Your own model bo putting your messager data into the folder 'data'. If you don't know how to do it there is a very nice guide here: https://www.zapptales.com/en/download-facebook-messenger-chat-history-how-to/
Delete text files in folder 'ngrok' and 'data'. 
Next put the Ngrok app in the folder 'ngrok'. You can get the app from here: https://ngrok.com/
Run Ngrok from command line with command '[directory]/ngrok http 5000
In chatbot.py add your facebook developer data and also reddit credentials. For Reddit: https://praw.readthedocs.io/en/latest/
                                                                            For Facebook: https://developers.facebook.com/
You will need to create your own facebook app on https://developers.facebook.com/ and also Page on https://Facebook.com/. 
Connect your app with the script and facebook page.
Run chatbot.py
