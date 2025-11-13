//This tests variables as array elements in JS

const axios = require('axios');

const tag = 'tag2';

axios.get('https://petstore.swagger.io/v2/pets', {
    params: {
        tags: ["tag1",tag],
        limit: 10
    }
});