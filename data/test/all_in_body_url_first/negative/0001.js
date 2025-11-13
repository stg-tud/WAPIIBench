// Method before url should fail
const axios = require('axios');
axios.request({
    method: 'get',
    url: 'https://petstore.swagger.io/v2/pets',
    params: {
        tags: ['tag1', 'tag2'],
        limit: 10
    }
});