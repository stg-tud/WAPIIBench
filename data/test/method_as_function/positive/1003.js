const axios = require('axios');
const tag = 1337
axios.post("https://petstore.swagger.io/v2/pets", {
    name: "name",
    tag: `${tag}`
});