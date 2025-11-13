// simple request
const axios = require('axios');
axios.request({
    url: 'https://petstore.swagger.io/v2/pets',
    method: 'get',
    params: {
        tags: ['tag1', 'tag2'],
        limit: 10
    }
});