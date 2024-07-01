import axios from 'axios';
import React, { useState } from 'react';
import './App.css';

const App = () => {
  const [maxTokens, setMaxTokens] = useState(100);
  const [generatedText, setGeneratedText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleGenerateText = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/generate', {
        max_tokens: maxTokens,
      });
      setGeneratedText(response.data.generated_text);
    } catch (error) {
      console.error('Error generating text:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App min-h-screen bg-amber-50 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="bg-amber-700 text-white p-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">Shakespeare Text Generator</h1>
        </div>
        <div className="p-6">
          <div className="mb-4">
            <label htmlFor="maxTokens" className="block text-sm font-medium text-gray-700 mb-1">
              Maximum Tokens:
            </label>
            <input
              type="number"
              id="maxTokens"
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
            />
          </div>
          <button
            onClick={handleGenerateText}
            disabled={loading}
            className="w-full bg-amber-600 text-white py-2 px-4 rounded-md hover:bg-amber-700 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-opacity-50 transition duration-150 ease-in-out disabled:opacity-50"
          >
            {loading ? 'Generating...' : 'Generate'}
          </button>
          <div className="mt-6 bg-amber-50 p-4 rounded-md min-h-[200px]">
            {generatedText && (
              <p style={{ fontSize: '1em', whiteSpace: 'pre-wrap' }}>{generatedText}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
