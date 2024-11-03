import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';
import './marketAnalysis.css';
import Legend from './Legend';


// Set up the default icon for markers
L.Marker.prototype.options.icon = L.icon({
    iconUrl: markerIcon,
    shadowUrl: markerShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
});

const center = [-37.8136, 144.9631]; // Melbourne's approximate center

function MarketAnalysis() {
    const [AvgSalesSuburbs, setAvgSalesSuburbs] = useState([]);
    const [AvgRentSuburbs, setAvgRentSuburbs] = useState([]);

    // Fetch avg-sales suburb data
    useEffect(() => {
        async function fetchAvgSalesData() {
            try {
                const response = await fetch('/avgSalesSuburbData.json');
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const suburbData = await response.json();
                setAvgSalesSuburbs(suburbData);
            } catch (error) {
                console.error("Failed to fetch avg sales suburb data:", error);
            }
        }

        async function fetchAvgRentData() {
            try {
                const response = await fetch('/avgRentSuburbsWithCoordinates.json');
                const data = await response.json();
                setAvgRentSuburbs(data);
            } catch (error) {
                console.error("Error fetching avg-rent data:", error);
            }
        }

        fetchAvgSalesData();
        fetchAvgRentData();
    }, []);

    // Define color based on rent values
    const getRentMarkerColor = (rent) => {
        if (rent > 400) return 'red';
        if (rent > 300) return 'orange';
        return 'green';
    };

    return (
        <div className="market-analysis-container">
            
            {/* Avg-Sales Map */}
            <div className="map-container">
                <h2>Average Sales Price of Properties by Suburb</h2>
                <MapContainer center={center} zoom={12} className="leaflet-container">
                    <TileLayer
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        attribution="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
                    />
                    {AvgSalesSuburbs.map((suburb, index) => (
                        suburb.coordinates && suburb.coordinates.lat && suburb.coordinates.lng ? (
                            <Marker
                                key={index}
                                position={[suburb.coordinates.lat, suburb.coordinates.lng]}
                            >
                                <Popup>
                                    <strong>{suburb.suburb}</strong><br />
                                    Average Sales Price: ${suburb.average_price.toLocaleString()}
                                </Popup>
                            </Marker>
                        ) : null
                    ))}
                </MapContainer>
            </div>

            {/* Avg-Rent Map */}
            <div className="map-container">
            
                <h2>Average Rent Price of Properties by Suburb</h2>
                <Legend />
                <MapContainer center={center} zoom={12} className="leaflet-container2">
                    <TileLayer
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        attribution="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
                    />
                    {AvgRentSuburbs.map((suburb, index) => (
                        suburb.coordinates && suburb.coordinates.lat && suburb.coordinates.lng ? (
                            <Marker
                                key={index}
                                position={[suburb.coordinates.lat, suburb.coordinates.lng]}
                                icon={L.divIcon({
                                    className: 'custom-marker',
                                    html: `<div style="background-color: ${getRentMarkerColor(suburb.average_rent)}; width: 20px; height: 20px; border-radius: 50%;"></div>`
                                })}
                            >
                                <Popup>
                                    <strong>{suburb.Suburb}</strong><br />
                                    Average Rent: ${suburb.average_rent.toFixed(2)}
                                </Popup>
                            </Marker>
                        ) : null
                    ))}
                </MapContainer>
            </div>
        </div>
    );
}

export default MarketAnalysis;
