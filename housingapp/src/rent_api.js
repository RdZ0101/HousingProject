const express = require('express');
const cors = require('cors');
const fs = require('fs');
const csv = require('csv-parser');
const app = express();
const PORT = 5001;
const csvFilePath = '../../Rent_1BF_Final.csv';

app.use(cors());

let suburbs = [];


fs.createReadStream(csvFilePath)
    .pipe(csv({ headers: ['Suburb', 'Postcode', 'Mar 2000', 'Jun 2000', 'Sep 2000', 'Dec 2000',], skipLines: 1 }))
    .on('data', (row) => {
        const suburb = row['Suburb']?.trim();
        if (suburb) {
            suburbs.push(suburb);
        }
    })
    .on('end', () => {
        console.log('CSV file successfully processed');

        app.listen(PORT, () => {
            console.log(`Server running on http://localhost:${PORT}`);
        });
    })
    .on('error', (error) => {
        console.error('Error reading CSV file:', error);
    });

// API endpoint to get suburb suggestions
app.get('/api/suburbs', (req, res) => {
    const query = req.query.query?.toLowerCase() || '';
    const suggestions = suburbs
        .filter(sub => typeof sub === 'string' && sub.toLowerCase().startsWith(query));
    res.json(suggestions);
});
