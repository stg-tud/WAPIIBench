// simple get request
const axios = require('axios');
axios.request({
    url:'https://petstore.swagger.io/v2/pets',
    method: 'get'
});