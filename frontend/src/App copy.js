import React from 'react';
import ReactDOM from 'react-dom';
import ShakespeareTextGenerator from './components/ShakespeareTextGenerator'; // Ajusta la ruta según tu estructura de directorios

const App = () => {
  return (
    <div className="App">
      <ShakespeareTextGenerator />
    </div>
  );
};

export default App; // Asegúrate de exportar el componente App por defecto

// Luego, renderiza App en el elemento root del DOM
ReactDOM.render(<App />, document.getElementById('root'));
