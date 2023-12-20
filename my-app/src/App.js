import logo from './logo.svg';
import './App.css';
import {useState} from "react";


const createOptions = (body) => {
    return {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({user_input: body})

    }
};

function App() {
    const [message, setMessage] = useState("");
    const [spamResult, setSpamResult] = useState("");


    return (
        <div className="App">
            <header className="App-header">
                <h2>Input for possible fraudulent text</h2>
                <h2>{spamResult}</h2>
                <input type="text" value={message} onChange={(message) => setMessage(message.target.value)}
                       className="form-control"/>
                <button onClick={async () => {
                    let responseFromFetch = await fetch("http://127.0.0.1:5000/test", createOptions(message))
                        .then(response => response.json())
                    setSpamResult(responseFromFetch.prediction);
                }
                }> Test text
                </button>

            </header>
        </div>
    );
}

export default App;
