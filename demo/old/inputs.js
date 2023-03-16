
function initKeys() {
    const keys = {
        " ": false,
        "`": false,
        "Shift": false,
        "Control": false,
        "Alt": false,
        "F1": false,
        "F2": false,
        "F3": false,
        "F4": false,
        "F5": false,
        "F6": false,
        "F7": false,
        "F8": false,
        "1": false,
        "2": false,
        "3": false,
        "4": false,
        "5": false,
        "6": false,
        "7": false,
        "8": false,
        "9": false,
        "10": false,
        "q": false,
        "w": false,
        "e": false,
        "r": false,
        "t": false,
        "y": false,
        "u": false,
        "i": false,
        "o": false,
        "p": false,
        "a": false,
        "s": false,
        "d": false,
        "f": false,
        "g": false,
        "h": false,
        "j": false,
        "k": false,
        "l": false,
        "z": false,
        "x": false,
        "c": false,
        "v": false,
        "b": false,
        "n": false,
        "m": false,
    }

    const keyChangeHandler = (evt) => {
        let toggle = evt.type === "keydown" ? true : false;

        keys[evt.key] = toggle;

    };

    document.addEventListener('keydown', (evt) => keyChangeHandler(evt));
    document.addEventListener('keyup', (evt) => keyChangeHandler(evt));

    return keys;

}

class InputHandler {

    constructor() {

        this.keys = initKeys();
        
    }

}

export default InputHandler;
