import React from 'react';
import './App.css';
import Home from './Home';
import SalesPred from './SalesPredictions';
import RentPred from './RentPredictions';
import Footer from './Footer'; // Import the footer
import { Routes, Route } from 'react-router-dom';
import NavBar from './NavBar';

function App() {
  return (
    <div className="App">
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/SalesPredictions" element={<SalesPred />} />
        <Route path="/RentPredictions" element={<RentPred />} />
        {/* Add any other routes here */}
      </Routes>
      <Footer /> 
    </div>
  );
}

export default App;
