const express = require('express');
const cors = require('cors');
const fs = require('fs');
const csv = require('csv-parser');
const app = express();
const PORT = 5000;
const csvFilePath = '../../Dataset/Melbourne_housing_FULL.csv';

app.use(cors());

// Load and store unique suburbs
let uniqueSuburbs = new Set();

fs.createReadStream(csvFilePath)
    .pipe(csv())
    .on('data', (row) => {
    uniqueSuburbs.add(row.Suburb);
    })
    .on('end', () => {
    uniqueSuburbs = Array.from(uniqueSuburbs);
    console.log('CSV file processed, unique suburbs loaded');
    });

// API endpoint to get suburb suggestions based on user input
app.get('/api/suburbs', (req, res) => {
    const query = req.query.query?.toLowerCase() || '';
    const suggestions = uniqueSuburbs.filter(suburb => suburb.toLowerCase().startsWith(query));
    res.json(suggestions);
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
