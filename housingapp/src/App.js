import React from 'react'; 
import './App.css';
import Home from './Home';
import SalesPred from './SalesPredictions';
import RentPred from './RentPredictions';
import GeneratedSalesPrediction from './GeneratedSalesPrediction';
import GeneratedRentPred from './GeneratedRentPredictions'; 
import MarketAnalysis from './MarketAnalysis'; 
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
		  <Route path="/GeneratedSalesPrediction" element={<GeneratedSalesPrediction />} />
          <Route path="/GeneratedRentPredictions" element={<GeneratedRentPred />} />
          <Route path="/MarketAnalysis" element={<MarketAnalysis />} /> 
        </Routes>
      </div>
      <Footer /> 
    </div>
  );
}

export default App;
