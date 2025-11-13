// This was supposed to test empty data/config when no parameters are defined, but since you can add a Content-Type header to delete request, the test was changed to test for empty data *and* config
const axios = require('axios');
const id = 42
axios.delete(`https://petstore.swagger.io/v2/pets/${id}`, {}, {});