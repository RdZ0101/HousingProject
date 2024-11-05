import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import axios from 'axios';
import Papa from 'papaparse';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function RoomCountLineChart({ userPostcode, houseType }) {
    const [chartData, setChartData] = useState(null);
    const [postcodeToSuburbMap, setPostcodeToSuburbMap] = useState({});
    const [suburb, setSuburb] = useState('');

    useEffect(() => {
        fetch(`${process.env.PUBLIC_URL}/Melbourne_housing_FULL.csv`)
            .then(response => response.text())
            .then(csvData => {
                Papa.parse(csvData, {
                    header: true,
                    complete: (results) => {
                        const mapping = {};
                        results.data.forEach(row => {
                            if (row.Suburb && row.Postcode) {
                                mapping[row.Postcode] = row.Suburb;
                            }
                        });
                        setPostcodeToSuburbMap(mapping);
                        setSuburb(mapping[userPostcode] || userPostcode);
                    },
                });
            });
    }, [userPostcode]);

    function reverseMapHouseType(value) {
        switch (value) {
            case 0:
                return 'house';
            case 1:
                return 'unit';
            case 2:
                return 'townhouse';
            default:
                return 'unknown';
        }
    }

    useEffect(() => {
        async function fetchPredictions() {
            const house = reverseMapHouseType(houseType);
            const roomCounts = [1, 2, 3, 4, 5];

            try {
                const response = await axios.post('http://127.0.0.1:8000/predict_room_prices', {
                    postcode: userPostcode,
                    house_type: houseType,
                    room_counts: roomCounts,
                });

                const predictions = roomCounts.map((rooms) => response.data.predictions[rooms]);

                setChartData({
                    labels: roomCounts,
                    datasets: [
                        {
                            label: `Predicted Price for ${house} in ${suburb}`,
                            data: predictions,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        },
                    ],
                });
            } catch (error) {
                console.error("Error fetching predictions:", error);
            }
        }

        if (suburb) {
            fetchPredictions();
        }
    }, [userPostcode, houseType, suburb]);

    return (
        chartData ? <Line data={chartData} /> : <p>Loading chart...</p>
    );
}

export default RoomCountLineChart;
