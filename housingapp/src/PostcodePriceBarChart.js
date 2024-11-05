import React, { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import axios from 'axios';
import Papa from 'papaparse';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function PostcodePriceBarChart({ userPostcode, userRooms, userHouseType }) {
    const [chartData, setChartData] = useState(null);
    const [postcodeToSuburbMap, setPostcodeToSuburbMap] = useState({});

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
                    },
                });
            });
    }, []);

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
            const house = reverseMapHouseType(userHouseType);
            const postcodes = [userPostcode, '3067', '3042', '3206', '3147', '3183', '3103', '3104'];

            try {
                const response = await axios.post('http://127.0.0.1:8000/predict_postcode_prices', {
                    postcode: postcodes,
                    house_type: userHouseType,
                    room_counts: userRooms,
                });
                const predictions = postcodes.map((postcode) => response.data.predictions[postcode]);

                const labels = postcodes.map(pc => postcodeToSuburbMap[pc] || pc);

                setChartData({
                    labels: labels,
                    datasets: [
                        {
                            label: `Predicted Price for ${userRooms} Room(s) - ${house}`,
                            data: predictions,
                            backgroundColor: postcodes.map((pc) =>
                                pc === userPostcode ? 'rgba(255, 99, 132, 0.6)' : 'rgba(54, 162, 235, 0.6)'
                            ),
                        },
                    ],
                });
            } catch (error) {
                console.error("Error fetching predictions:", error);
            }
        }

        fetchPredictions();
    }, [userPostcode, userRooms, userHouseType, postcodeToSuburbMap]);

    const options = {
        responsive: true,
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Predicted Prices by Suburb' },
        },
    };

    return chartData ? <Bar data={chartData} options={options} /> : <p>Loading...</p>;
}

export default PostcodePriceBarChart;
