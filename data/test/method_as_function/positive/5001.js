// This tests complex objects as array elements
const axios = require('axios');
axios.post('https://petstore.swagger.io/v2/customers', {
    name: 'name',
    addresses: [
        {
            city: "Darmstadt",
            house: {
                street: "street",
                number: 1
            }
        },
        {
            city: "Berlin",
            house: {
                street: "street",
                number: 1
            }
        }
    ]
});