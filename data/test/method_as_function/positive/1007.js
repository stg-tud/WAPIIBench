// This tests for escaped backslashes inside quotes
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/pets', {
    name: 'na\\\\me',
    tag: 'tag'
});