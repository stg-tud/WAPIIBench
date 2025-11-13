// This tests objects as attributes, with not all required attributes
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/foods', {
    name: 'name',
    nutrition: {
        protein: 10,
        fat: 20,
        calories: 40
     },
});