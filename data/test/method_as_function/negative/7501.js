// This tests requests with empty request body
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/customers/johndoe/ping', {
    params: {
        priority: 20
    }
});