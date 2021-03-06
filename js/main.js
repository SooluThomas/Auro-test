(function() {
    var Message;
    Message = function(arg) {
        this.text = arg.text, this.message_side = arg.message_side;
        this.draw = function(_this) {
            return function() {
                var $message;
                $message = $($('.message_template').clone().html());
                $message.addClass(_this.message_side).find('.text').html(_this.text);
                $('.messages').append($message);
                return setTimeout(function() {
                    return $message.addClass('appeared');
                }, 0);
            };
        }(this);
        return this;
    };
    $(function() {
        var lastSend;
        var getMessageText, message_side;
        message_side = 'right';
        getMessageText = function() {
            var $message_input;
            $message_input = $('.message_input');
            return $message_input.val();
        };
        sendMessage = function(text) {
            var $messages, message;
            if (text.trim() === '') {
                return;
            }
            $('.message_input').val('');
            $messages = $('.messages');

            message_side = message_side === 'left' ? 'left' : 'left';
            message = new Message({
                text: text,
                message_side: message_side
            });
            message.draw();
            return $messages.animate({
                scrollTop: $messages.prop('scrollHeight')
            }, 300);
        };
        receiveMessage = function(text) {
            var $messages, message;
            if (text.trim() === '') {
                return;
            }
            $('.message_input').val('');
            $messages = $('.messages');

            message_side = message_side === 'right' ? 'right' : 'right';
            message = new Message({
                text: text,
                message_side: message_side
            });
            message.draw();
            return $messages.animate({
                scrollTop: $messages.prop('scrollHeight')
            }, 300);
        };
        // $('.send_message').click(function(e) {
        //     lastSend = $('.message_input').val();
        //     return receiveMessage(getMessageText());
        // });
        $('.message_input').keyup(function(e) {
            lastSend = $('.message_input').val();
            if (e.which === 13) {
                 var user_data=lastSend;
                console.log(user_data);
                RequestAPI(user_data);
                return receiveMessage(getMessageText());

            }
        });
        sendMessage('Hello User! What would you like to know?');
        $('#send_button').on('click',function(){
            var user_data=lastSend;
            console.log(user_data);
            RequestAPI(user_data);
        });
    });
}.call(this))

function RequestAPI(request_data){
    myURL = 'https://auro-api.herokuapp.com/api/?q=' + request_data;
        var api_reply;
        var ourRequest = new XMLHttpRequest();
        ourRequest.open('GET', myURL);
        ourRequest.onreadystatechange=function(){
            if(this.readyState==4 && this.status==200){
                api_reply=this.responseText;
                // console.log(api_reply);
                sendMessage(api_reply);
            }
        }
        ourRequest.open('GET', myURL, true);
        ourRequest.send();
}
