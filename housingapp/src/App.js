import React from 'react'; 
import './App.css';
import Home from './Home';
import SalesPred from './SalesPredictions';
import RentPred from './RentPredictions';
import GenSalesPred from './GeneratedSalesPrediction';
import GeneratedRentPred from './GeneratedRentPredictions'; 
import Footer from './Footer';
import { Routes, Route } from 'react-router-dom';
import NavBar from './NavBar';

function App() {
  return (
    <div className="app-container">
      <NavBar />
      <div className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/SalesPredictions" element={<SalesPred />} />
          <Route path="/RentPredictions" element={<RentPred />} />
          <Route path="/GeneratedSalesPrediction" element={<GenSalesPred />} />
          <Route path="/GeneratedRentPredictions" element={<GeneratedRentPred />} /> 
        </Routes>
      </div>
      <Footer /> 
    </div>
  );
}

export default App;