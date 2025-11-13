const axios = require('axios');
const requestBody = {
    name: 'name',
    tag: 'tag'
};
axios.post('https://petstore.swagger.io/v2/pets', requestBody);