import React, { useState } from 'react';
import './App.css';

function App() {
    const [inputText, setInputText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async () => {
        if (!inputText.trim()) return;
    
        setLoading(true);
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText }),
                });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error:', error);
            setResult({ error: "Failed to get prediction" });
        }
        setLoading(false);
    };

    return (
        <div className="App">
            <h1>Sentiment Analysis</h1>
            <div className="input-box">
                <label htmlFor="text-input">Enter Text:</label>
                <textarea
                    id="text-input"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    rows="4"
                    disabled={loading}
                />
                <button 
                    onClick={handleSubmit}
                    disabled={loading || !inputText.trim()}
                >
                    {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                </button>
            </div>

            {result && (
                <div className="result-box">
                    {result.error ? (
                        <p className="error">Error: {result.error}</p>
                    ) : (
                        <>
                            <h2>Analysis Result</h2>
                            <p>Sentiment: <strong>{result.sentiment}</strong></p>
                            <p>Confidence: {result.confidence.toFixed(4)}</p>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
