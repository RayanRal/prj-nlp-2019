"""Trident code below"""

import requests
from stem import Signal
from stem.control import Controller

PROXIES = {
    'http' : 'socks5h://localhost:9150',
    'https' : 'socks5h://localhost:9150'
}

HEADERS = {
    'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Encoding': 'br, gzip, deflate',
    'Connection': 'keep-alive'
}


def init():
    session = requests.session(headers=HEADERS)
    session.proxies = PROXIES
    
    return session


def shift(session):
    session.cookies.clear()
    with Controller.from_port(port = 9151) as controller:
        controller.authenticate()
        if controller.is_newnym_available():
            controller.signal(Signal.NEWNYM)
        else:
            sec_left = controller.get_newnym_wait()
            print("Error: you seem to be shifting identities too often.")
            print("Try again in %s seconds" % sec_left)