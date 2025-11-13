//This tests advanced whitespaces in arrays

const axios = require('axios');

const tag = 'tag2';

axios.get('https://petstore.swagger.io/v2/pets', {
    params: {
        tags: ["tag1",       tag,
            'tag3', 'tag4',     'tag5'
        ],
        limit: 10
    }
});