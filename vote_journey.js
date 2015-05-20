// ==UserScript==
// @name         vote_journey
// @namespace     http://your.homepage/
// @version      0.1
// @description  enter something useful
// @author       You
// @match         http://www.enet.com.cn/enews/zhuanti/2015/vote_journey/?from=singlemessage&isappinstalled=0
// ==/UserScript==

//unsafeWindow.alert = function alert(message) {console.log(message)};
tpv(268,1);
var waitTime = 30000 + Math.round(Math.random() * 1000);
console.log(waitTime);

var result = document.getElementById('vid_21');
var rank = result.parentNode;
rank = rank.childNodes[1].childNodes[0];
console.log(result.innerHTML);
console.log(rank.innerHTML);
document.title = rank.innerHTML + '名 ' + result.innerHTML;

setInterval(function () {
    window.location.reload();
}, waitTime);
