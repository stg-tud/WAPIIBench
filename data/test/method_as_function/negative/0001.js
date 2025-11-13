const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/pets', {
    tags: ['tag1', 'tag2'],
    limit: 10
});