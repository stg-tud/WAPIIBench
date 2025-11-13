// This tests objects as attributes, with invalid attribute data types
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/foods', {
    name: 'name',
    nutrition: {
        protein: "10",
        fat: 20,
        carbs: 30,
        calories: 40
     },
});