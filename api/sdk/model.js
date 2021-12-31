Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@AchmadBurhan10 
AchmadBurhan10
/
UAS_SC_18_Achmad-Burhanudin_41419010005
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
UAS_SC_18_Achmad-Burhanudin_41419010005/api/index.js /
@AchmadBurhan10
AchmadBurhan10 Update index.js
Latest commit 3f0ea51 2 hours ago
 History
 1 contributor
78 lines (69 sloc)  1.8 KB
   
var express = require('express');
var r = express.Router();

// load pre-trained model
const model = require('./sdk/model.js');

// Bot Setting
const TelegramBot = require('node-telegram-bot-api');
const token = '5073546476:AAG4rie_ytVRpRfHLYNLt0cDz_YcjDidZAE'
const bot = new TelegramBot(token, {polling: true});


// bots
bot.onText(/\/start/, (msg) => { 
    console.log(msg)
    bot.sendMessage(
        msg.chat.id,
        `halo selamat datang ! ${msg.chat.first_name}, welcome...\n
        click /prediksi`
    );   
});
// input requires x1 , x2 dan x3
state = 0;
bot.onText(/\/prediksi/, (msg) => { 
    bot.sendMessage(
        msg.chat.id,
        `Masukan nilai x1|x2|x3 contohnya 8|8|8`
    );
    state = 1;
});

bot.on('message', (msg) => (
    if(state == 1){
        s = msg.text.split("|");
        x1 = s[0]
        x2 = s[1]
        x3 = s[2]
        model.prediksi(
            [ 
                parsefloat(s[0]), // string float
                parsefloat(s[1]),
                parsefloat(s[2])
           
            ]
        ).them((jres)=>{
            bot.sendmessage(
                msg.chat.id,
                'Nilai y1 yang diprediksi adalah ${jres[0])'
            );  
            bot.sendmessage(
                msg.chat.id,
                'Nilai y2 yang diprediksi adalah ${jres[1])'
            );
            bot.sendmessage(
                msg.chat.id,
                'Nilai y3 yang diprediksi adalah ${jres[2])'
             );
        })
    }else{
        state = 0
    }
})

// routers
r.get('prediksi/:x1/:x2/:x3', function(req, res, next) {
    model.prediksi(
        [
            parsefloat(req.params.x1),// string float
            parsefloat(req.params.x2),
            parsefloat(req.params.x3)
        ]    
    }.then((jres)=>{
        res.json(jres);
    })
});

module.exports = r:
